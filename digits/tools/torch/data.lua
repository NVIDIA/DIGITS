-- Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.
require 'torch' -- torch
require 'nn' -- provides a normalization operator
require 'utils' -- various utility functions
require 'hdf5' -- import HDF5 now as it is unsafe to do it from a worker thread
local threads = require 'threads' -- for multi-threaded data loader
check_require('image') -- for color transforms

package.path = debug.getinfo(1, "S").source:match[[^@?(.*[\/])[^\/]-$]] .."?.lua;".. package.path

require 'logmessage'
local ffi = require 'ffi'

local tdsIsInstalled, tds = pcall(function() return check_require('tds') end)

-- enable shared serialization to speed up Tensor passing between threads
threads.Threads.serialization('threads.sharedserialize')

----------------------------------------------------------------------

function copy (t) -- shallow-copy a table
    if type(t) ~= "table" then return t end
    local meta = getmetatable(t)
    local target = {}
    for k, v in pairs(t) do target[k] = v end
    setmetatable(target, meta)
    return target
end

local function all_keys(cursor_,key_,op_)
    return coroutine.wrap(
        function()
            local k = key_
            local v
            repeat
                k,v = cursor_:get(k,op_ or MDB.NEXT)
                if k then
                    coroutine.yield(k,v)
                end
            until (not k)
        end)
end

-- HSV Augmentation
-- Parameters:
-- @param im (tensor): input image
-- @param augHSV (table): standard deviations under {H,S,V} keys with float values.
local augmentHSV = function(im_rgb, augHSV)
    -- Fair augHSV standard deviation values are {H=0.02,S=0.04,V=0.08}
    local im_hsv = image.rgb2hsv(im_rgb)
    if augHSV.H >0 then
        -- We do not need to account for overflow because every number wraps around (1.1=2.1=3.1,etc)
        -- We add a round value (+ 1) to prevent an underflow bug (<0 becomes glitchy)
        im_hsv[1] = im_hsv[1]+(1 + torch.normal(0, augHSV.S))
    end
    if augHSV.S >0 then
        im_hsv[2] = im_hsv[2]+torch.normal(0, augHSV.S)
        im_hsv[2].image.saturate(im_hsv[2]) -- bound saturation between 0 and 1
    end
    if augHSV.V >0 then
        im_hsv[3] = im_hsv[3]+torch.normal(0, augHSV.V)
        im_hsv[3].image.saturate(im_hsv[3]) -- bound value between 0 and 1
    end
    return image.hsv2rgb(im_hsv)
end

-- Scale and Rotation augmentation (warping)
-- Parameters:
-- @param im (tensor): input image
-- @param augRot (float): extremes of random rotation, uniformly distributed between
-- @param augScale (float): the standard deviation of the extra scaling factor
local warp = function(im, augRot, augScale)
    -- A nice function of scale is 0.05 (stddev of scale change),
    -- and a nice value for ration is a few degrees or more if your dataset allows for it

    local width = im:size()[3]
    local height = im:size()[2]

    -- Scale <0=zoom in(+rand crop), >0=zoom out
    local scale_x = 0
    local scale_y = 0
    local move_x = 0
    local move_y = 0
    if augScale > 0 then
        scale_x = torch.normal(0, augScale) -- normal distribution
        -- Given a zoom in or out, we move around our canvas.
        scale_y = scale_x -- keep aspect ratio the same
        move_x = torch.uniform(-scale_x, scale_x)
        move_y = torch.uniform(-scale_y, scale_y)
    end

    -- Angle of rotation
    local rot_angle = torch.uniform(-augRot,augRot) -- (degrees) uniform distribution [-augRot : augRot)

    -- x/y grids
    local grid_x = torch.ger( torch.ones(height), torch.linspace(-1-scale_x,1+scale_x,width) )
    local grid_y = torch.ger( torch.linspace(-1-scale_y,1+scale_y,height), torch.ones(width) )

    local flow = torch.FloatTensor()
    flow:resize(2,height,width)
    flow:zero()

    -- Apply scale
    flow_scale = torch.FloatTensor()
    flow_scale:resize(2,height,width)
    flow_scale[1] = grid_y
    flow_scale[2] = grid_x
    flow_scale[1]:add(1+move_y):mul(0.5) -- move ~[-1 1] to ~[0 1]
    flow_scale[2]:add(1+move_x):mul(0.5) -- move ~[-1 1] to ~[0 1]
    flow_scale[1]:mul(height-1)
    flow_scale[2]:mul(width-1)
    flow:add(flow_scale)

    if augRot > 0 then
        -- Apply rotation through rotation matrix
        local flow_rot = torch.FloatTensor()
        flow_rot:resize(2,height,width)
        flow_rot[1] = grid_y * ((height-1)/2) * -1
        flow_rot[2] = grid_x * ((width-1)/2) * -1
        view = flow_rot:reshape(2,height*width)
        function rmat(deg)
          local r = deg/180*math.pi
          return torch.FloatTensor{{math.cos(r), -math.sin(r)}, {math.sin(r), math.cos(r)}}
        end

        local rotmat = rmat(rot_angle)
        local flow_rotr = torch.mm(rotmat, view)
        flow_rot = flow_rot - flow_rotr:reshape( 2, height, width )
        flow:add(flow_rot)
    end

    return image.warp(im, flow, 'bilinear', false)
end

-- Adds noise to the image
-- Parameters:
-- @param im (tensor): input image
-- @param augNoise (float): the standard deviation of the white noise
local addNoise = function(im, augNoise)
    -- AWGN:
    -- torch.randn makes noise with mean 0 and variance 1 (=stddev 1)
    --  so we multiply the tensor with our augNoise factor, that has a linear relation with
    --  the standard deviation (but the variance will be increased quadratically).
    return torch.add(im, torch.randn(im:size()):float()*augNoise)
end

-- Quadrilateral rotation (through flipping and/or transposing)
-- Parameters:
-- @param im (tensor): input image
-- @param rotFlag (int): rotation indices (0 1 2 3)=(0,90CW,270CW,180CW)
local rot90 = function(im, rotFlag)
    local rot
    if rotFlag == 2 then
        rot = im:transpose(2,3) --switch X and Y dimensions
        rot = image.vflip(rot)
        return rot
    elseif rotFlag == 3 then
        rot = im:transpose(2,3) --switch X and Y dimensions
        rot = image.hflip(rot)
        return rot
    elseif rotFlag == 4 then -- vflip+hflip=180 deg rotation
        rot = image.hflip(im)
        rot = image.vflip(rot)
        return rot
    end
    return im -- no rotation: return in place
end

-- PreProcessing of a single image
-- Parameters:
-- @param im (tensor): input image
-- @param train (boolean): distinguishes training phase from testing phase
-- @param meanTensor (tensor): mean image that will be used for mean subtraction if supplied
-- @param augOpt (table): structure containing several augmentation options.
local PreProcess = function(im, train, meanTensor, augOpt)

    -- Do the HSV augmentation on the [0 255] image
    if augOpt.augHSV then
        if (augOpt.augHSV.H > 0) or (augOpt.augHSV.S > 0)  or (augOpt.augHSV.V > 0) then
            assert(im:size(1)==3, 'Cannot mix HSV augmentation without the image having 3 channels.')
            im:div(255) -- HSV augmentation requires a [0:1] range
            im = augmentHSV(im, augOpt.augHSV)
            im:mul(255)
            -- Note: an RGB-spaced image is returned (as was the input)
        end
    end

    -- Noise addition
    if augOpt.augNoise and (augOpt.augNoise > 0) then
        -- Note: augNoise is multiplied by 255 because the input assumes [0 1] image range
        --  and our current range is [0 255]
        im = addNoise(im, augOpt.augNoise*255)
    end

    -- Note :do any augmentation that directly changes pixel values before mean subtraction
    if meanTensor then
        if meanTensor:nDimension() > 1 then
            -- mean image subtraction - test whether sizes match
            assert(meanTensor:nDimension() == im:nDimension(), 'Tensor dimension mismatch')
            size_match = true
            for i=1,im:nDimension() do
                if meanTensor:size(i) ~= im:size(i) then size_match = false end
            end
            if not size_match then
                -- we need to resize the mean image to the dimensions of the input sample
                -- it is highly inefficient to do so therefore mean pixel subtraction is
                -- preferred when working with variable image sizes
                logmessage.display(1,'resizing mean: use mean pixel subtraction for better performance')
                meanTensor = image.scale(meanTensor, im:size(3), im:size(2))
            end
        end
        for i=1,meanTensor:size(1) do
            -- mean *pixel* subtraction => meanTensor[i] is a scalar
            -- mean *image* subtraction => meanTensor[i] is a tensor
            im[i]:csub(meanTensor[i])
        end
    end

    if augOpt.augFlip then
        if (augOpt.augFlip == 'fliplr' or  augOpt.augFlip == 'fliplrud') and torch.random(2)==1 then
            im = image.hflip(im)
        end
        if (augOpt.augFlip == 'flipud' or  augOpt.augFlip == 'fliplrud') and torch.random(2)==1 then
            im = image.vflip(im)
        end
    end

    if augOpt.augQuadRot then
        if augOpt.augQuadRot == 'rot90' then -- (0, 90 or 270)
            im = rot90(im, torch.random(3))
        elseif augOpt.augQuadRot == 'rot180' and torch.random(2)==1 then -- (0 or 180)
            im = rot90(im, 4)
        elseif augOpt.augQuadRot == 'rotall' then -- (0, 90, 180 or 270)
            im = rot90(im,  torch.random(4))
        end
    end

    if augOpt.augRot and augOpt.augScale and ((augOpt.augRot >0) or (augOpt.augScale > 0)) then
        im = warp(im, augOpt.augRot, augOpt.augScale)
    end

    if augOpt.crop and augOpt.crop.use then
        if train == true then
            --During training we will crop randomly
            local valueY = math.ceil(torch.FloatTensor.torch.uniform() * augOpt.crop.Y)
            local valueX = math.ceil(torch.FloatTensor.torch.uniform() * augOpt.crop.X)
            im = image.crop(im, valueX-1, valueY-1, valueX + augOpt.crop.len-1, valueY + augOpt.crop.len-1)
        else
            --for validation we will crop at center
            im = image.crop(im, augOpt.crop.X-1, augOpt.crop.Y-1, augOpt.crop.X + augOpt.crop.len-1, augOpt.crop.Y + augOpt.crop.len-1)
        end
    end

    return im
end

-- Loading label definitions
local loadLabels = function(labels_file)
    local Classes = {}
    i=0

    local file = io.open(labels_file)
    if file then
        for line in file:lines() do
            i=i+1
            -- labels file might contain Windows line endings
            -- or other whitespaces => trim leading and trailing
            -- spaces here
            Classes[i] = trim(line)
        end
        return Classes
    else
        return nil -- nil indicates that file not present
    end
end

-- Loading mean tensor
local loadMean = function(mean_file, use_mean_pixel)
    local meanTensor
    local mean_im = image.load(mean_file,nil,'byte'):type('torch.FloatTensor'):contiguous()
    if use_mean_pixel then
        mean_of_mean = torch.FloatTensor(mean_im:size(1))
        for i=1,mean_im:size(1) do
            mean_of_mean[i] = mean_im[i]:mean()
        end
        meanTensor = mean_of_mean
    else
        meanTensor = mean_im
    end
    return meanTensor
end

local function pt(t)
    for k,v in pairs(t) do
        print(k,v)
    end
end

-- Meta class
LMDBSource = {e=nil, t=nil, d=nil, c=nil}

function LMDBSource:new(lighningmdb, path)
    local paths = require('paths')
    assert(paths.dirp(path), 'DB Directory '.. path .. ' does not exist')
    logmessage.display(0,'opening LMDB database: ' .. path)
    local self = copy(LMDBSource)
    self.lightningmdb = lighningmdb
    self.e = self.lightningmdb.env_create()
    local LMDB_MAP_SIZE = 1099511627776 -- 1 TB
    self.e:set_mapsize(LMDB_MAP_SIZE)
    local flags = lightningmdb.MDB_RDONLY + lightningmdb.MDB_NOTLS
    local db, err = self.e:open(path, flags, 0664)
    if not db then
        -- unable to open Database => this might be due to a permission error on
        -- the lock file so we will try again with MDB_NOLOCK. MDB_NOLOCK is safe
        -- in this process as we are opening the database in read-only mode.
        -- However if another process is writing into the database we might have a
        -- concurrency issue - note that this shouldn't happen in DIGITS since the
        -- database is written only once during dataset creation
        logmessage.display(0,'opening LMDB database failed with error: "' .. err .. '". Trying with MDB_NOLOCK')
        flags = bit.bor(flags, lighningmdb.MDB_NOLOCK)
        -- we need to close/re-open the LMDB environment
        self.e:close()
        self.e = self.lightningmdb.env_create()
        self.e:set_mapsize(LMDB_MAP_SIZE)
        db, err = self.e:open(path, flags ,0664)
        if not db then
            error('opening LMDB database failed with error: ' .. err)
        end
    end
    self.total = self.e:stat().ms_entries
    self.t = self.e:txn_begin(nil, self.lightningmdb.MDB_RDONLY)
    self.d = self.t:dbi_open(nil,0)
    self.c = self.t:cursor_open(self.d)
    self.path = path
    return self, self.total
end

function LMDBSource:close()
    logmessage.display(0,'closing LMDB database: ' .. self.path)
    self.total = 0
    self.c:close()
    self.e:dbi_close(self.d)
    self.t:abort()
    self.e:close()
end

function LMDBSource:get(key)
    if key then
        v = self.t:get(self.d,key,self.lightningmdb.MDB_FIRST)
    else
        k,v = self.c:get(nil,self.lightningmdb.MDB_NEXT)
    end
    return v
end

-- Meta class
DBSource = {mean = nil, ImageChannels = 0, ImageSizeY = 0, ImageSizeX = 0, total=0,
            augOpt={}, subtractMean=true,
            train=false, classification=false}

-- Derived class method new
-- Creates a new instance of a database
-- Parameters:
-- @param backend (string): 'lmdb' or 'hdf5'
-- @param db_path (string): path to database
-- @param labels_db_path (string): path to labels database, or nil
-- @param meanTensor (tensor): mean tensor to use for mean subtraction
-- @param isTrain (boolean): whether this is a training database (e.g. mirroring not applied to validation database)
-- @param shuffle (boolean): whether samples should be shuffled
-- @param classification (boolean): whether this is a classification task
function DBSource:new (backend, db_path, labels_db_path, meanTensor, isTrain, shuffle, classification)
    local self = copy(DBSource)
    local paths = require('paths')

    if backend == 'lmdb' then
        check_require('pb')
        self.datum = pb.require"datum"
        local lightningmdb_lib=check_require("lightningmdb")
        self.lightningmdb = _VERSION=="Lua 5.2" and lightningmdb_lib or lightningmdb
        self.lmdb_data, self.total = LMDBSource:new(self.lightningmdb, db_path)
        self.keys = self:lmdb_getKeys()
        if #self.keys ~= self.total then
            logmessage.display(2, 'thr-id=' .. __threadid .. ' will be disabled - failed to initialize DB - #keys=' .. #self.keys .. ' #records='.. self.total)
            return nil
        end
        -- do we have a separate database for labels/targets?
        if labels_db_path~='' then
            self.lmdb_labels, total_labels = LMDBSource:new(self.lightningmdb, labels_db_path)
            assert(self.total == total_labels, "Number of records="..self.total.." does not match number of labels=" ..total_labels)
        end
        -- LMDB-specific functions
        self.getSample = self.lmdb_getSample
        self.close = self.lmdb_close
        self.reset = self.lmdb_reset
        -- read first entry to check image dimensions
        local v = self.lmdb_data:get(self.keys[1])
        local msg = self.datum.Datum():Parse(v)
        self.ImageChannels = msg.channels
        self.ImageSizeX = msg.width
        self.ImageSizeY = msg.height
    elseif backend == 'hdf5' then
        assert(classification, "hdf5 only supports classification yet")
        assert(labels_db_path == '', "hdf5 separate label DB not implemented yet")
        check_require('hdf5')
        -- list.txt contains list of HDF5 databases --
        logmessage.display(0,'opening HDF5 database: ' .. db_path)
        list_path = paths.concat(db_path, 'list.txt')
        local file = io.open(list_path)
        assert(file,"unable to open "..list_path)
        -- go through all DBs in list.txt and store required info
        self.dbs = {}
        self.total = 0
        for line in file:lines() do
            local fileName = paths.concat(db_path, paths.basename(line))
            -- get number of records
            local myFile = hdf5.open(fileName,'r')
            local dim = myFile:read('/data'):dataspaceSize()
            local n_records = dim[1]
            self.ImageChannels = dim[2]
            self.ImageSizeY = dim[3]
            self.ImageSizeX = dim[4]
            myFile:close()
            -- store DB info
            self.dbs[#self.dbs + 1] = {
                path = fileName,
                records = n_records
            }
            self.total = self.total + n_records
        end
        -- which DB is currently being used (initially set to nil to defer
        -- read to first call to self:nextBatch)
        self.db_id = nil
        -- cursor points to current record in current DB
        self.cursor = nil
        -- set pointers to HDF5-specific functions
        self.getSample = self.hdf5_getSample
        self.close = self.hdf5_close
        self.reset = self.hdf5_reset
    end

    if meanTensor ~= nil then
        self.mean = meanTensor
        assert(self.ImageChannels == self.mean:size()[1])
        if self.mean:nDimension() > 1 then
            -- mean matrix subtraction
            assert(self.ImageSizeY == self.mean:size()[2])
            assert(self.ImageSizeX == self.mean:size()[3])
        end
    end

    logmessage.display(0,'Image channels are ' .. self.ImageChannels .. ', Image width is ' .. self.ImageSizeX .. ' and Image height is ' .. self.ImageSizeY)

    self.train = isTrain
    self.shuffle = shuffle
    self.classification = classification

    -- Initialize with no dataset augmentation
    self.augOpt.augFlip = 'none'
    self.augOpt.augQuadRot = 'none'
    self.augOpt.augRot = 0
    self.augOpt.augScale = 0
    self.augOpt.augNoise = 0
    self.augOpt.augHSV = {H=0, S=0, V=0}
    self.augOpt.ConvertColor = 'none'
    self.augOpt.crop = {use=false, Y=-1, X=-1, len=-1}

    return self
end

-- Derived class method setDataAugmentation
-- Parameters:
-- @param augOpt (table): structure containing several augmentation options. Processing differs between train and test state.
function DBSource:setDataAugmentation(augOpt)
    if self.train == true then
        self.augOpt = augOpt -- Copy all augmentation options
        if self.augOpt.crop.use then
            self.augOpt.crop.Y = self.ImageSizeY - self.augOpt.crop.len + 1
            self.augOpt.crop.X = self.ImageSizeX - self.augOpt.crop.len + 1
        end
    else -- Validation:
        -- For validation, only copy certain options (constructor default is no augmentation)
        -- So we are doing very selective copying of the augmentation options
        if augOpt.crop.use then
            self.augOpt.crop = augOpt.crop
            self.augOpt.crop.Y = math.floor((self.ImageSizeY - self.augOpt.crop.len)/2) + 1
            self.augOpt.crop.X = math.floor((self.ImageSizeX - self.augOpt.crop.len)/2) + 1
        end
    end
end

-- Derived class method inputTensorShape
-- This returns the shape of the input samples in the database
-- There is an assumption that all input samples have the same shape
function DBSource:inputTensorShape()
    local shape = torch.Tensor(3)
    shape[1] = self.ImageChannels
    shape[2] = self.ImageSizeY
    shape[3] = self.ImageSizeX
    return shape
end

-- Derived class method lmdb_getKeys
function DBSource:lmdb_getKeys ()
    local Keys
    if tdsIsInstalled then
        -- use tds.Vec() to allocate memory outside of Lua heap
        Keys = tds.Vec()
    else
        -- if tds is not installed, use regular table (and Lua memory allocator)
        Keys = {}
    end
    local i=0
    local key=nil
    for k,v in all_keys(self.lmdb_data.c,nil,self.lightningmdb.MDB_NEXT) do
        i=i+1
        Keys[i] = k
        key = k
    end
    return Keys
end

-- Decode a protobuf-encoded datum object
-- Parameters:
-- @param datum The object to decode
-- @return t (decoded tensor), label (scalar label within datum)
function DBSource:decodeDatum(datum)
    local msg = self.datum.Datum():Parse(datum)
    local t, label

    if msg.float_data then
        -- flat vector of floats
        -- initialize tensor with data from msg.float_data table
        t = torch.FloatTensor(msg.float_data)
    else
        -- create tensor to copy message data to.
        -- note the 1 extra byte to copy null terminator
        local x = torch.ByteTensor(#msg.data+1):contiguous()
        ffi.copy(torch.data(x), msg.data)
        if msg.encoded then
            -- x is an encoded image (.png or .jpg)
            t = image.decompress(x, msg.channels,'byte'):float()
        else
            -- x is an unencoded CHW matrix
            -- drop null-terminator
            x = x:narrow(1, 1, x:storage():size() - 1)
            -- view as CHW matrix of floats
            x = x:view(msg.channels, msg.height, msg.width):float()
            -- is this an RGB image?
            if self.ImageChannels == 3 then
                -- unencoded color images are stored in BGR order => we need to swap blue and red channels (BGR->RGB)
                t = torch.FloatTensor(msg.channels, msg.height, msg.width)
                t[1] = x[3]
                t[2] = x[2]
                t[3] = x[1]
            else
                t = x
            end
        end
    end

    -- note: the scalar label is ignored if there is a dedicated DB for labels
    label = msg.label

    return t, label
end


-- Derived class method getSample (LMDB flavour)
function DBSource:lmdb_getSample(shuffle, idx)
    if shuffle then
        idx = math.max(1,torch.ceil(torch.rand(1)[1] * self.total))
    end

    local key = self.keys[idx]
    local v = self.lmdb_data:get(key)
    assert(key~=nil, "lmdb read nil key at idx="..idx)
    assert(v~=nil, "lmdb read nil value at idx="..idx.." key="..key)

    -- decode protobuf-encoded Datum object
    local im, label = self:decodeDatum(v)

    if self.lmdb_labels then
        -- read label from label database
        local v = self.lmdb_labels:get(key)
        label = self:decodeDatum(v)
    end

    return im, label
end

-- Derived class method getSample (HDF5 flavour)
function DBSource:hdf5_getSample(shuffle)
    if not self.db_id or self.cursor>self.dbs[self.db_id].records then
        --local a = torch.Timer()
        --local m = a:time().real
        self.db_id = self.db_id or 0
        assert(self.db_id < #self.dbs, "Trying to read more records than available")
        self.db_id = self.db_id + 1
        local myFile = hdf5.open(self.dbs[self.db_id].path, 'r')
        self.hdf5_data = myFile:read('/data'):all()
        self.hdf5_labels = myFile:read('/label'):all()
        myFile:close()
        -- make sure number of entries match number of labels
        assert(self.hdf5_data:size()[1] == self.hdf5_labels:size()[1], "data/label mismatch")
        self.cursor = 1
        --print(string.format("hdf5 data load - elapsed time1: %.6f\n", a:time().real - m))
    end

    local idx
    if shuffle then
        idx = math.max(1,torch.ceil(torch.rand(1)[1] * self.dbs[self.db_id].records))
    else
        idx = self.cursor
    end
    local im = self.hdf5_data[idx]
    channels = im:size()[1]
    label = self.hdf5_labels[idx]
    self.cursor = self.cursor + 1
    return im, label
end

-- Derived class method nextBatch
-- Parameters:
-- @param batchSize (int): Number of samples to load
-- @param idx (int): Current index within database
function DBSource:nextBatch (batchSize, idx)

    local images, labels

    -- this function creates a tensor that has similar
    -- shape to that of the provided sample plus one
    -- dimension (batch dimension)
    local function createBatchTensor(sample, batchSize)
        local t
        if type(sample) == 'number' then
            t = torch.Tensor(batchSize)
        else
            shape = sample:size():totable()
            -- add 1 dimension (batchSize)
            table.insert(shape, 1, batchSize)
            t = torch.Tensor(torch.LongStorage(shape))
        end
        return t
    end

    for i=1,batchSize do
        -- get next sample
        local im, label = self:getSample(self.shuffle, idx + i - 1)

        if self.classification then
            -- label is index from array and Lua array indices are 1-based
            label = label + 1
        end

        -- preprocess
        im = PreProcess(im, self.train, self.mean, self.augOpt)


        -- create batch tensors if not already done
        if not images then
            images = createBatchTensor(im, batchSize)
        end
        if not labels then
            labels = createBatchTensor(label, batchSize)
        end

        images[i] = im
        labels[i] = label
    end

    return images, labels
end

-- Derived class method to reset cursor (HDF5 flavour)
function DBSource:hdf5_reset ()
    self.db_id = nil
    self.cursor = nil
end

-- Derived class method to get total number of Records
function DBSource:totalRecords ()
    return self.total;
end

-- Derived class method close (LMDB flavour)
function DBSource:lmdb_close ()
    self.lmdb_data:close()
    if self.lmdb_labels then
        self.lmdb_labels:close()
    end
end

-- Derived class method close (HDF5 flavour)
function DBSource:hdf5_close ()
end

-- Meta class
DataLoader = {}

-- Derived class method new
-- Creates a new instance of a database
-- Parameters:
-- @param numThreads (int): number of reader threads to create
-- @param package_path (string): caller package path
-- @param backend (string): 'lmdb' or 'hdf5'
-- @param db_path (string):  path to database
-- @param labels_db_path (string): path to labels database, or nil
-- @param mean (Tensor): mean tensor for mean image
-- @param isTrain (boolean): whether this is a training database (e.g. mirroring not applied to validation database)
-- @param shuffle (boolean): whether samples should be shuffled
-- @param classification (boolean): whether this is a classification task
function DataLoader:new (numThreads, package_path, backend, db_path, labels_db_path, mean, isTrain, shuffle, classification)
    local self = copy(DataLoader)
    self.backend = backend
    if self.backend == 'hdf5' then
        -- hdf5 wrapper does not support multi-threaded reader yet
        numThreads = 1
    end
    -- create pool of threads
    self.numThreads = numThreads
    self.threadPool = threads.Threads(
        self.numThreads,
        function(threadid)
            -- inherit package path from main thread
            package.path = package_path
            require('data')
            -- executes in reader thread, variables are local to this thread
            db = DBSource:new(backend, db_path, labels_db_path,
                              mean,
                              isTrain,
                              shuffle,
                              classification
                              )
        end
    )
    -- use non-specific mode
    self.threadPool:specific(false)
    return self
end

function DataLoader:getInfo()
    local datasetSize
    local inputTensorShape

    -- switch to specific mode so we can specify which thread to add job to
    self.threadPool:specific(true)
    -- we need to iterate here as some threads may not have a valid DB
    -- handle (as happens when opening too many concurrent instances)
    for i=1,self.numThreads do
        self.threadPool:addjob(
                   i, -- thread to add job to
                   function()
                       -- executes in reader thread, return values passed to
                       -- main thread through following function
                       if db then
                           return db:totalRecords(), db:inputTensorShape()
                       else
                           return nil, nil
                       end
                   end,
                   function(totalRecords, shape)
                       -- executes in main thread
                       datasetSize = totalRecords
                       inputTensorShape = shape
                   end
                   )
        self.threadPool:synchronize()
        if datasetSize then
            break
        end
    end
    -- return to non-specific mode
    self.threadPool:specific(false)
    return datasetSize, inputTensorShape
end

-- Schedule next data loader batch
-- Parameters:
-- @param batchSize (int): Number of samples to load
-- @param dataIdx (int): Current index in database
-- @param dataTable (table): Table to store data into
function DataLoader:scheduleNextBatch(batchSize, dataIdx, dataTable)
    -- send reader thread a request to load a batch from the training DB
    local backend = self.backend
    self.threadPool:addjob(
                function()
                    if backend=='hdf5' and dataIdx==1 then
                        db:reset()
                    end
                    -- executes in reader thread
                    if db then
                        inputs, targets =  db:nextBatch(batchSize, dataIdx)
                        return batchSize, inputs, targets
                    else
                        return batchSize, nil, nil
                    end
                end,
                function(batchSize, inputs, targets)
                    -- executes in main thread
                    dataTable.batchSize = batchSize
                    dataTable.inputs = inputs
                    dataTable.outputs = targets
                end
            )
end

-- returns whether data loader is able to accept more jobs
function DataLoader:acceptsjob()
    return self.threadPool:acceptsjob()
end

-- wait until next data loader job completes
function DataLoader:waitNext()
    -- wait for next data loader job to complete
    self.threadPool:dojob()
    -- check for errors in loader threads
    if self.threadPool:haserror() then -- check for errors
        self.threadPool:synchronize() -- finish everything and throw error
    end
end

-- free data loader resources
function DataLoader:close()
    -- switch to specific mode so we can specify which thread to add job to
    self.threadPool:specific(true)
    for i=1,self.numThreads do
        self.threadPool:addjob(
                    i,
                    function()
                        if db then
                            db:close()
                        end
                    end
                )
    end
    -- return to non-specific mode
    self.threadPool:specific(false)
end

-- Set dataset augmentation parameters (calls setDataAugmentation() method of all DB intances)
-- Parameters:
-- @param augOpt (table): structure containing several augmentation options.
function DataLoader:setDataAugmentation(augOpt)
    -- Switch to specific mode so we can specify which thread to add job to
    self.threadPool:specific(true)
    for i=1,self.numThreads do
        self.threadPool:addjob(
                    i,
                    function()
                        if db then
                            db:setDataAugmentation(augOpt)
                        end
                    end
                )
    end
    -- return to non-specific mode
    self.threadPool:specific(false)
end

return{
    loadLabels = loadLabels,
    loadMean= loadMean,
    PreProcess = PreProcess
}

