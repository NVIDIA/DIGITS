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
    if subtractMean == 'yes' then
        for i=1,channels do
            y[{ i,{},{} }]:add(-mean[i])
        end
    end
    if mirror == 'yes' and torch.FloatTensor.torch.uniform() > 0.49 then
        y = image.hflip(y)
    end
    if crop == 'yes' then

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
    if use_mean_pixel == 'yes' then
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
DBSource = {e=nil, t=nil, d=nil, c=nil, mean = nil, ImageChannels = 0, ImageSizeY = 0, ImageSizeX = 0, total=0, mirror='no', crop='no', croplen=0, cropY=0, cropX=0, subtractMean='yes', train=false}

-- Derived class method new
function DBSource:new (db_name, backend, mirror, crop, croplen, mean_t, subtractMean, isTrain, shuffle, path_api)
    local self = copy(DBSource)

    if backend == 'lmdb' then
        check_require('pb')
        self.datum = pb.require"datum"
        local lightningmdb_lib=check_require("lightningmdb")
        self.lightningmdb = _VERSION=="Lua 5.2" and lightningmdb_lib or lightningmdb
        self.e = self.lightningmdb.env_create()
        local LMDB_MAP_SIZE = 1099511627776 -- 1 TB
        self.e:set_mapsize(LMDB_MAP_SIZE)
        self.e:open(db_name,self.lightningmdb.MDB_RDONLY + self.lightningmdb.MDB_NOTLS,0664)
        self.total = self.e:stat().ms_entries
        self.t = self.e:txn_begin(nil,self.lightningmdb.MDB_RDONLY)
        self.d = self.t:dbi_open(nil,0)
        self.c = self.t:cursor_open(self.d)
        self.keys = self:lmdb_getKeys()
        self.getSample = self.lmdb_getSample
        self.close = self.lmdb_close
        self.reset = self.lmdb_reset
    elseif backend == 'hdf5' then
        check_require('hdf5')
        -- list.txt contains list of HDF5 databases --
        local paths = require('paths')
        list_path = paths.concat(db_name, 'list.txt')
        local file = io.open(list_path)
        assert(file,"unable to open "..list_path)
        -- go through all DBs in list.txt and store required info
        self.dbs = {}
        self.total = 0
        for line in file:lines() do
            db_path = paths.concat(db_name, paths.basename(line))
            -- get number of records
            local myFile = hdf5.open(db_path,'r')
            local dim = myFile:read('/data'):dataspaceSize()
            local n_records = dim[1]
            myFile:close()
            -- store DB info
            self.dbs[#self.dbs + 1] = {
                path = db_path,
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

    self.mean = mean_t["mean"]
    -- image channel, height and width details are extracted from mean.jpeg file. If mean.jpeg file is not present then probably the below three lines of code needs to be changed to provide hard-coded values.
    self.ImageChannels = mean_t["channels"]
    self.ImageSizeY = mean_t["height"]
    self.ImageSizeX = mean_t["width"]

    logmessage.display(0,'Loaded train image details from the mean file: Image channels are ' .. self.ImageChannels .. ', Image width is ' .. self.ImageSizeY .. ' and Image height is ' .. self.ImageSizeX)

    self.mirror = mirror
    self.crop = crop
    self.croplen = croplen
    self.subtractMean = subtractMean
    self.train = isTrain
    self.shuffle = shuffle

    if crop == 'yes' then
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
    for k,v in all_keys(self.c,nil,self.lightningmdb.MDB_NEXT) do
        i=i+1
        Keys[i] = k
        key = k
    end
    return Keys
end

-- Derived class method getSample (LMDB flavour)
function DBSource:lmdb_getSample(shuffle)

    if shuffle then
        key_idx = torch.ceil(torch.rand(1)[1] * self.total)
        key = self.keys[key_idx]
        v = self.t:get(self.d,key,self.lightningmdb.MDB_FIRST)
    else
        k,v = self.c:get(k,self.lightningmdb.MDB_NEXT)
    end

    local total = self.ImageChannels*self.ImageSizeY*self.ImageSizeX
    -- Tensor allocations inside loop consumes little more execution time. So allocated "x" outiside with double size of an image and inside loop if any encoded image is encountered with bytes size more than Tensor size, then the Tensor is resized appropriately.
    local x = torch.ByteTensor(total*2):contiguous() -- some times length of JPEG files are more than total size. So, "x" is allocated with more size to ensure that data is not truncated while copying.
    local x_size = total * 2 -- This variable is just to avoid the calls to tensor's size() i.e., x:size(1)
    local temp_ptr = torch.data(x) -- raw C pointer using torchffi


    --local a = torch.Timer()
    --local m = a:time().real
    local msg = self.datum.Datum():Parse(v)

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

    return y, msg.channels, msg.label

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
    return y, channels, label
end

-- Derived class method nextBatch
function DBSource:nextBatch (batchsize)

    local Images
    if self.crop == 'yes' then
        Images = torch.Tensor(batchsize, self.ImageChannels, self.croplen, self.croplen)
    else
        Images = torch.Tensor(batchsize, self.ImageChannels, self.ImageSizeY, self.ImageSizeX)
    end
    local Labels = torch.Tensor(batchsize)

    local i=0

    --local mean_ptr = torch.data(self.mean)
    local image_ind = 0

    for i=1,batchsize do
        -- get next sample
        y, channels, label = self:getSample(self.shuffle)

        Images[i] = PreProcess(y, self.mean, self.subtractMean, channels, self.mirror, self.crop, self.train, self.cropY, self.cropX, self.croplen)

        Labels[i] = tonumber(label) + 1
    end
    return Images, Labels
end

-- Derived class method to reset cursor (LMDB flavour)
function DBSource:lmdb_reset ()

    self.c:close()
    self.e:dbi_close(self.d)
    self.t:abort()
    self.t = self.e:txn_begin(nil,self.lightningmdb.MDB_RDONLY)
    self.d = self.t:dbi_open(nil,0)
    self.c = self.t:cursor_open(self.d)
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
    self.total = 0
    self.c:close()
    self.e:dbi_close(self.d)
    self.t:abort()
    self.e:close()
end

-- Derived class method close (HDF5 flavour)
function DBSource:hdf5_close (batchsize)
end

return{
    loadLabels = loadLabels,
    loadMean= loadMean,
    PreProcess = PreProcess
}

