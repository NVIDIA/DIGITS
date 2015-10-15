-- Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
require 'torch' -- torch
require 'nn' -- provides a normalization operator
require 'utils' -- various utility functions
check_require 'image' -- for color transforms

package.path = debug.getinfo(1, "S").source:match[[^@?(.*[\/])[^\/]-$]] .."?.lua;".. package.path

require 'logmessage'
local ffi = require 'ffi'

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
            local k = key_,v
            repeat
                k,v = cursor_:get(k,op_ or MDB.NEXT)
                if k then
                    coroutine.yield(k,v)
                end
            until (not k)
        end)
end

local PreProcess = function(y, mean, subtractMean, channels, mirror, crop, train, cropY, cropX, croplen)
    if subtractMean then
        for i=1,channels do
            y[{ i,{},{} }]:add(-mean[i])
        end
    end
    if mirror and torch.FloatTensor.torch.uniform() > 0.49 then
        y = image.hflip(y)
    end
    if crop then

        if train == true then
            --During training we will crop randomly
            local valueY = math.ceil(torch.FloatTensor.torch.uniform()*cropY)
            local valueX = math.ceil(torch.FloatTensor.torch.uniform()*cropX)
            y = image.crop(y, valueX-1, valueY-1, valueX+croplen-1, valueY+croplen-1)

        else
            --for validation we will crop at center
            y = image.crop(y, cropX-1, cropY-1, cropX+croplen-1, cropY+croplen-1)
        end
    end
    return y
end

--Loading label definitions

local loadLabels = function(labels_file)
    local Classes = {}
    i=0

    local file = io.open(labels_file)
    if file then
        for line in file:lines() do
            i=i+1
            Classes[i] = line
        end
        return Classes
    else
        return nil -- nil indicates that file not present
    end
end

--loading mean tensor
local loadMean = function(mean_file, use_mean_pixel)
    local mean_t = {}
    local mean_im = image.load(mean_file,nil,'byte'):type('torch.FloatTensor'):contiguous()
    mean_t["channels"] = mean_im:size(1)
    mean_t["height"] = mean_im:size(2)
    mean_t["width"] = mean_im:size(3)
    if use_mean_pixel then
        mean_of_mean = torch.FloatTensor(mean_im:size(1))
        for i=1,mean_im:size(1) do
            mean_of_mean[i] = mean_im[i]:mean()
        end
        mean_t["mean"] = mean_of_mean
    else
        mean_t["mean"] = mean_im
    end
    return mean_t
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
    self.e:open(path, self.lightningmdb.MDB_RDONLY + self.lightningmdb.MDB_NOTLS,0664)
    self.total = self.e:stat().ms_entries
    self.t = self.e:txn_begin(nil, self.lightningmdb.MDB_RDONLY)
    self.d = self.t:dbi_open(nil,0)
    self.c = self.t:cursor_open(self.d)
    return self, self.total
end

function LMDBSource:reset()
    self.c:close()
    self.e:dbi_close(self.d)
    self.t:abort()
    self.t = self.e:txn_begin(nil,self.lightningmdb.MDB_RDONLY)
    self.d = self.t:dbi_open(nil,0)
    self.c = self.t:cursor_open(self.d)
end

function LMDBSource:close()
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
            mirror=false, crop=false, croplen=0, cropY=0, cropX=0, subtractMean=true,
            train=false, classification=false}

-- Derived class method new
-- Creates a new instance of a database
-- Parameters:
--  backend (string): 'lmdb' or 'hdf5'
--  db_path (string): path to database
--  labels_db_path (string): path to labels database, or nil
--  mirror (boolean): whether samples must be mirrored
--  crop (boolean): whether samples must be cropped
--  croplen (int): crop length, if crop==true
--  mean_t (table): table containing mean tensor for mean image, if subtractMean==true
--  subtractMean (boolean): whether mean image should be subtracted from samples
--  isTrain (boolean): whether this is a training database (e.g. mirroring not applied to validation database)
--  shuffle (boolean): whether samples should be shuffled
--  classification (boolean): whether this is a classification task
function DBSource:new (backend, db_path, labels_db_path, mirror, crop, croplen, mean_t, subtractMean, isTrain, shuffle, classification)
    local self = copy(DBSource)
    local paths = require('paths')

    if backend == 'lmdb' then
        check_require('pb')
        self.datum = pb.require"datum"
        local lightningmdb_lib=check_require("lightningmdb")
        self.lightningmdb = _VERSION=="Lua 5.2" and lightningmdb_lib or lightningmdb
        self.lmdb_data, self.total = LMDBSource:new(self.lightningmdb, db_path)
        self.keys = self:lmdb_getKeys()
        -- do we have a separate database for labels/targets?
        if labels_db_path~='' then
            self.lmdb_labels, total_labels = LMDBSource:new(self.lightningmdb, labels_db_path)
            assert(self.total == total_labels, "Number of records="..self.total.." does not match number of labels=" ..total_labels)
            -- read first entry to check label dimensions
            local v = self.lmdb_labels:get(self.keys[1])
            local msg = self.datum.Datum():Parse(v)
            -- assume row vector 1xN labels
            assert(msg.height == 1, "label height=" .. msg.height .. " not supported")
            self.label_width = msg.width
        else
            -- assume scalar label (e.g. classification)
            self.label_width = 1
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
        -- assume scalar label (e.g. classification)
        self.label_width = 1
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

    if subtractMean then
        self.mean = mean_t["mean"]
        assert(self.ImageChannels == self.mean:size()[1])
        assert(self.ImageSizeY == self.mean:size()[2])
        assert(self.ImageSizeX == self.mean:size()[3])
    end

    logmessage.display(0,'Image channels are ' .. self.ImageChannels .. ', Image width is ' .. self.ImageSizeY .. ' and Image height is ' .. self.ImageSizeX)

    self.mirror = mirror
    self.crop = crop
    self.croplen = croplen
    self.subtractMean = subtractMean
    self.train = isTrain
    self.shuffle = shuffle
    self.classification = classification

    if self.classification then
        assert(self.label_width == 1, 'expect scalar labels for classification tasks')
    end

    if crop then
        if self.train == true then
            self.cropY = self.ImageSizeY - croplen + 1
            self.cropX = self.ImageSizeX - croplen + 1
        else
            self.cropY = math.floor((self.ImageSizeY - croplen)/2) + 1
            self.cropX = math.floor((self.ImageSizeX - croplen)/2) + 1
        end
    end

    return self
end

-- Derived class method lmdb_getKeys
function DBSource:lmdb_getKeys ()
    local Keys = {}
    local i=0
    local key=nil
    for k,v in all_keys(self.lmdb_data.c,nil,self.lightningmdb.MDB_NEXT) do
        i=i+1
        Keys[i] = k
        key = k
    end
    return Keys
end

-- Derived class method getSample (LMDB flavour)
function DBSource:lmdb_getSample(shuffle)
    local label

    if shuffle then
        key_idx = torch.ceil(torch.rand(1)[1] * self.total)
    else
        key = nil
    end

    v = self.lmdb_data:get(key)

    local total = self.ImageChannels*self.ImageSizeY*self.ImageSizeX
    -- Tensor allocations inside loop consumes little more execution time. So allocated "x" outiside with double size of an image and inside loop if any encoded image is encountered with bytes size more than Tensor size, then the Tensor is resized appropriately.
    local x = torch.ByteTensor(total*2):contiguous() -- some times length of JPEG files are more than total size. So, "x" is allocated with more size to ensure that data is not truncated while copying.
    local x_size = total * 2 -- This variable is just to avoid the calls to tensor's size() i.e., x:size(1)
    local temp_ptr = torch.data(x) -- raw C pointer using torchffi


    --local a = torch.Timer()
    --local m = a:time().real
    local msg = self.datum.Datum():Parse(v)

    if not self.lmdb_labels then
        -- read label from sample database
        label = msg.label
    else
        -- read label from label database
        v = self.lmdb_labels:get(key)
        local label_msg = self.datum.Datum():Parse(v)
        label = torch.FloatTensor(label_msg.width):contiguous()
        for x=1,label_msg.width do
            label[x] = label_msg.float_data[x]
        end
    end

    if #msg.data > x_size then
        x:resize(#msg.data+1) -- 1 extra byte is required to copy zero terminator i.e., '\0', by ffi.copy()
        x_size = #msg.data
    end

    ffi.copy(temp_ptr, msg.data)
    --print(string.format("elapsed time1: %.6f\n", a:time().real - m))
    --m = a:time().real

    local y=nil
    if msg.encoded==true then
        y = image.decompress(x,msg.channels,'byte'):float()
    else
        y = x:narrow(1,1,total):view(msg.channels,msg.height,msg.width):float() -- using narrow() returning the reference to x tensor with the size exactly equal to total image byte size, so that view() works fine without issues
    end

    return y, label

end

-- Derived class method getSample (HDF5 flavour)
function DBSource:hdf5_getSample (shuffle)
    if not self.db_id or self.cursor>self.dbs[self.db_id].records then
        --local a = torch.Timer()
        --local m = a:time().real
        self.db_id = self.db_id or 0
        assert(self.db_id < #self.dbs, "Trying to read more records than available")
        self.db_id = self.db_id + 1
        local myFile = hdf5.open(self.dbs[self.db_id].path, 'r')
        self.hdf5_data = myFile:read('/data'):all()
        self.hdf5_labels = myFile:read('/label'):all()
        -- make sure number of entries match number of labels
        assert(self.hdf5_data:size()[1] == self.hdf5_labels:size()[1], "data/label mismatch")
        self.cursor = 1
        --print(string.format("hdf5 data load - elapsed time1: %.6f\n", a:time().real - m))
    end

    local idx
    if shuffle then
        idx = torch.ceil(torch.rand(1)[1] * self.dbs[self.db_id].records)
    else
        idx = self.cursor
    end
    y = self.hdf5_data[idx]
    channels = y:size()[1]
    label = self.hdf5_labels[idx]
    self.cursor = self.cursor + 1
    return y, label
end

-- Derived class method nextBatch
function DBSource:nextBatch (batchsize)

    local Images
    if self.crop then
        Images = torch.Tensor(batchsize, self.ImageChannels, self.croplen, self.croplen)
    else
        Images = torch.Tensor(batchsize, self.ImageChannels, self.ImageSizeY, self.ImageSizeX)
    end

    local Labels
    if self.label_width == 1 then
        Labels = torch.Tensor(batchsize)
    else
        Labels = torch.FloatTensor(batchsize, self.label_width)
    end

    local i=0

    --local mean_ptr = torch.data(self.mean)
    local image_ind = 0

    for i=1,batchsize do
        -- get next sample
        y, label = self:getSample(self.shuffle)

        if self.classification then
            -- label is index from array and Lua array indices are 1-based
            label = label + 1
        end

        Images[i] = PreProcess(y, self.mean, self.subtractMean, self.ImageChannels, self.mirror, self.crop, self.train, self.cropY, self.cropX, self.croplen)

        Labels[i] = label
    end
    return Images, Labels
end

-- Derived class method to reset cursor (LMDB flavour)
function DBSource:lmdb_reset ()
    self.lmdb_data:reset()
    if self.lmdb_labels then
        self.lmdb_labels:reset()
    end
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
function DBSource:lmdb_close (batchsize)
    self.lmdb_data:close()
    if self.lmdb_labels then
        self.lmdb_labels:close()
    end
end

-- Derived class method close (HDF5 flavour)
function DBSource:hdf5_close (batchsize)
end

return{
    loadLabels = loadLabels,
    loadMean= loadMean,
    PreProcess = PreProcess
}

