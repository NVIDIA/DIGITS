require 'torch'   -- torch
require 'image'   -- for color transforms
--require 'gfx.js'  -- to visualize the dataset
require 'nn'      -- provides a normalization operator
require "pb"

package.path = debug.getinfo(1, "S").source:match[[^@?(.*[\/])[^\/]-$]] .."?.lua;".. package.path

require 'logmessage'
local datum = pb.require"datum"
local ffi = require 'ffi'

local lightningmdb_lib=require "lightningmdb"
local lightningmdb = _VERSION=="Lua 5.2" and lightningmdb_lib or lightningmdb


----------------------------------------------------------------------

function copy (t) -- shallow-copy a table
  if type(t) ~= "table" then return t end
  local meta = getmetatable(t)
  local target = {}
  for k, v in pairs(t) do target[k] = v end
  setmetatable(target, meta)
  return target
end 


local function cursor_pairs(cursor_,batch_size,key_,op_)
  return coroutine.wrap(
    function()
      local k = key_,v
      local i =0
      repeat
        i=i+1
        k,v = cursor_:get(k,op_ or MDB.NEXT)
        if k then
          coroutine.yield(k,v)
        --[[else
          print('hi')
          k,v = cursor_:get(k,op_ or MDB.FIRST)
          coroutine.yield(k,v)]]-- 
        end
      until i==batch_size
    end)
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


local PreProcess = function(y, mean, subtractMean, channels, mirror, crop, train, cropX, cropY, croplen)
    if subtractMean == 'yes' then
        for i=1,channels  do
           y[{ i,{},{} }]:add(-mean[i])
        end
    end

    if mirror == 'yes' and torch.FloatTensor.torch.uniform() > 0.49 then
        y = image.hflip(y)
    end 
    if crop == 'yes' then

      if train == true then
        --During training we will crop randomly
        local valueX =  math.ceil(torch.FloatTensor.torch.uniform()*cropX)
        local valueY =  math.ceil(torch.FloatTensor.torch.uniform()*cropY)
        --y = image.crop(y, valueX, valueY, valueX+ImageSizeX-1, valueY+ImageSizeY-1)
        y = image.crop(y, valueY-1, valueX-1, valueY+croplen-1, valueX+croplen-1)

      else   
        --for validation we will crop at center
        --y = image.crop(y, cropX, cropY, cropX+ImageSizeX-1, cropY+ImageSizeY-1)
        y = image.crop(y, cropY-1, cropX-1, cropY+croplen-1, cropX+croplen-1)
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
    return nil    -- nil indicates that file not present
  end
end

--loading mean tensor
local loadMean = function(mean_file)
  return image.load(mean_file):type('torch.FloatTensor'):contiguous()
end
  

local function pt(t)
  for k,v in pairs(t) do
    print(k,v)
  end
end

-- Meta class
DBSource = {e=nil, t=nil, d=nil, c=nil, mean = nil, ImageChannels = 0, ImageSizeX = 0, ImageSizeY = 0, total=0, datum_t=datum, mirror='no', crop='no', croplen=0, cropX=0, cropY=0, subtractMean='yes', train=false}

-- Derived class method new
function DBSource:new (db_name, mirror, crop, croplen, mean_t, subtractMean, isTrain)
  --o = o or {}
  --setmetatable(o, self)
  local self = copy(DBSource)
  --self.__index = self
  self.mean = mean_t
  -- image channel, height and width details are extracted from mean.jpeg file. If mean.jpeg file is not present then probably the below three lines of code needs to be changed to provide hard-coded values.
  self.ImageChannels = mean_t:size(1)
  self.ImageSizeX = mean_t:size(2)
  self.ImageSizeY = mean_t:size(3)

  logmessage.display(0,'Loaded train image details from the mean file: Image channels are  ' .. self.ImageChannels .. ', Image width is ' .. self.ImageSizeX .. ' and Image height is ' .. self.ImageSizeY)

  self.e = lightningmdb.env_create()
  local LMDB_MAP_SIZE = 1099511627776  -- 1 TB
  self.e:set_mapsize(LMDB_MAP_SIZE)
  self.e:open(db_name,lightningmdb.MDB_RDONLY + lightningmdb.MDB_NOTLS,0664)
  self.total = self.e:stat().ms_entries
  self.t = self.e:txn_begin(nil,lightningmdb.MDB_RDONLY)
  self.d = self.t:dbi_open(nil,0)
  self.c = self.t:cursor_open(self.d)
  self.mirror = mirror
  self.crop = crop
  self.croplen = croplen
  self.subtractMean = subtractMean
  self.train = isTrain

  if crop == 'yes' then
    if self.train == true then
      self.cropX = self.ImageSizeX - croplen + 1
      self.cropY = self.ImageSizeY - croplen + 1
    else
      self.cropX = math.floor((self.ImageSizeX - croplen)/2) + 1
      self.cropY = math.floor((self.ImageSizeY - croplen)/2) + 1
    end
  end

  return self
end

-- Derived class method getKeys
function DBSource:getKeys ()

  local Keys = {}
  local i=0
  local key=nil
  for k,v in all_keys(self.c,nil,lightningmdb.MDB_NEXT) do
    i=i+1
    Keys[i] = k
    key = k
    -- xlua.progress(i,self.total)  
  end
  return Keys
end

function getFirstImage()
  return self.c:get(nil, MDB.FIRST)
end

-- Derived class method getKeys
function DBSource:getImgUsingKey(key)
  v = self.t:get(self.d,key,lightningmdb.MDB_FIRST)
  local msg = datum.Datum():Parse(v)
  local x=torch.ByteTensor(#msg.data):contiguous()
  local temp_ptr=torch.data(x)
  ffi.copy(temp_ptr, msg.data)
  local y=nil
  if msg.encoded==true then
    y = image.decompressJPG(x):float()
  else
    y=x:reshape(msg.channels,msg.height,msg.width):float()
  end
  
  local image_s = PreProcess(y, self.mean, self.subtractMean, msg.channels, self.mirror, self.crop, self.train, self.cropX, self.cropY, self.croplen)
 
  local label = msg.label
  return image_s,label
  
end

-- Derived class method nextBatch
function DBSource:nextBatch (batchsize)

  local Images
  if self.crop == 'yes' then
    Images = torch.Tensor(batchsize, self.ImageChannels, self.croplen, self.croplen)
  else
    --Images = torch.FloatTensor(batchsize, self.ImageChannels, self.ImageSizeX, self.ImageSizeY):contiguous()   -- This needs to be checked later. Is contiguous is necessary?
    Images = torch.Tensor(batchsize, self.ImageChannels, self.ImageSizeX, self.ImageSizeY)
  end
  local Labels = torch.Tensor(batchsize)

  local data=torch.data(Images)
  
  local i=0

  --local key=nil
  
--[[  
  if self.current == 0 then
    i=i+1
    local k,v=self.c:get(nil,lightningmdb.MDB_FIRST)
    local msg = datum.Datum():Parse(v)
    local x = torch.ByteTensor(#msg.data)
    local temp_ptr = torch.data(x) -- raw C pointer using torchffi
    ffi.copy(temp_ptr, msg.data)
    local y = x:reshape(msg.channels,msg.height,msg.width):float()
    Images[i] = PreProcess(y, self.mean)
    Labels[i] = tonumber(msg.label) + 1
  end
]]--
  local total = self.ImageChannels*self.ImageSizeX*self.ImageSizeY
  local x = torch.ByteTensor(total):contiguous()
  local temp_ptr = torch.data(x) -- raw C pointer using torchffi
  local mean_ptr = torch.data(self.mean)
  local image_ind = 0  

  for k,v in cursor_pairs(self.c,batchsize,nil,lightningmdb.MDB_NEXT) do
    i=i+1
--local a = torch.Timer()
--local m = a:time().real 
    local msg = datum.Datum():Parse(v)
    ffi.copy(temp_ptr, msg.data)
--print(string.format("elapsed time1: %.6f\n", a:time().real  - m))
--m = a:time().real

    local y=nil
    if msg.encoded==true then
      y = image.decompressJPG(x):float()
    else
      y=x:reshape(msg.channels,msg.height,msg.width):float()
    end

    --[[for ind=1,total do
      data[image_ind+ind] = temp_ptr[ind]-mean_ptr[ind]
    end]]--

    Images[i] = PreProcess(y, self.mean, self.subtractMean, msg.channels, self.mirror, self.crop, self.train, self.cropX, self.cropY, self.croplen)

    --print(string.format("elapsed time2: %.6f\n", a:time().real  - m))

    Labels[i] = tonumber(msg.label) + 1
    --image_ind = image_ind + total

    --key = k
  end
  return Images, Labels
end

-- Derived class method to reset cursor
function DBSource:reset ()

  self.c:close()
  self.e:dbi_close(self.d)
  self.t:abort()
  self.t = self.e:txn_begin(nil,lightningmdb.MDB_RDONLY)
  self.d = self.t:dbi_open(nil,0)
  self.c = self.t:cursor_open(self.d)
end

-- Derived class method to get total number of Records
function DBSource:totalRecords ()
  return self.total;
end

-- Derived class method close
function DBSource:close (batchsize)
  self.total = 0
  self.c:close()
  self.e:dbi_close(self.d)
  self.t:abort()
  self.e:close()
end


return{
    loadLabels = loadLabels,
    loadMean= loadMean
}

