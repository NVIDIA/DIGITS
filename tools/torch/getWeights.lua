-- Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'trepl'
require 'image'
require 'lfs'

local dir_path = debug.getinfo(1,"S").source:match[[^@?(.*[\/])[^\/]-$]]
if dir_path ~= nil then
    package.path = dir_path .."?.lua;".. package.path
end

require 'utils'
check_require 'hdf5'
require 'logmessage'
require 'Optimizer'

opt = lapp[[
-t,--threads (default 8) number of threads
-p,--type (default cuda) float or cuda
-d,--devid (default 1) device ID (if using CUDA)
-n,--network (string) Pretrained Model to be loaded
-y,--ccn2 (default no) should be 'yes' if ccn2 is used in network. Default : false
-s,--save (default .) save directory

--snapshot (string) Path to snapshot to load
--networkDirectory (default '') directory in which network exists
]]

torch.setnumthreads(opt.threads)

if opt.type =='cuda' then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.devid)
end

local utils = require 'utils'

local data = require 'data'

-- loads network model and performs various operations:
-- * reload weights from snapshot
-- * move model to CUDA or FLOAT
-- * adjust last layer to match number of labels (for classification networks)
function loadNetwork(dir, name, weightsFile, tensorType)
  package.path = paths.concat(dir, "?.lua") ..";".. package.path
  logmessage.display(0,'Loading network definition from ' .. paths.concat(dir, name))
  local parameters = {
      ngpus = (tensorType =='cuda') and 1 or 0,
  }
  if nn.DataParallelTable then
      -- set number of GPUs to use when deserializing model
      nn.DataParallelTable.deserializeNGPUs = parameters.ngpus
  end
  local network = require (name)(parameters)

  local model
  -- TODO: Support _Weights files

  -- the full model was saved
  assert(string.find(weightsFile, '_Model'))
  model = torch.load(weightsFile)
  network.model = model

  if tensorType =='cuda' then
      model:cuda()
  else
      model:float()
  end

  return network
end

local using_ccn2 = opt.ccn2

-- if ccn2 is used in network, then set using_ccn2 value as 'yes'
if ccn2 ~= nil then
    using_ccn2 = 'yes'
end

local weights_filename = opt.snapshot

local network = loadNetwork(opt.networkDirectory, opt.network, weights_filename, opt.type)

local filename = paths.concat(opt.save, 'weights.h5')
logmessage.display(0,'Saving visualization to ' .. filename)
local vis_db = hdf5.open(filename, 'w')

local layer_id = 1

function traverseModel(layer, link, chain)
  local weights = layer.weight
  local bias = layer.bias
  name = tostring(layer)

  tname = utils.stringToTensor(name)
  chain = chain .. "_" .. link

  if weights ~= nil then
      vis_db:write('/layers/'..chain..'/weights', weights:float())
  end

  if bias ~= nil then
      vis_db:write('/layers/'..chain..'/bias', bias:float())
  end

  vis_db:write('/layers/'..chain..'/layer_id', utils.stringToTensor(tostring(layer_id)))
  vis_db:write('/layers/'..chain..'/name', tname )

  layer_id = layer_id + 1

  if layer.modules then
    for i=1,#layer.modules do
      child = layer:get(i)
      link = link + 1
      traverseModel(child,link,chain)
    end
  end

end

traverseModel(network.model, 0, "")
vis_db:close()
