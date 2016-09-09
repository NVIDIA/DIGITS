-- Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
require 'cutorch'
require 'cudnn'
require 'cunn'
require 'image'
require 'nn'
require 'os'
require 'pl'
require 'torch'

local dir_path = debug.getinfo(1,"S").source:match[[^@?(.*[\/])[^\/]-$]]
if dir_path ~= nil then
    package.path = dir_path .."?.lua;".. package.path
end
require 'utils'
check_require 'hdf5'
require 'logmessage'
require 'Optimizer'
require 'nnhelpers'
local _ = require('moses')

opt = lapp[[
-p,--type (default cuda) float or cuda
-n,--network (string) Pretrained Model to be loaded
--weights (string) model weights to load
--networkDirectory (default '') directory in which network exists
-s,--save (default .) save directory
--mean_file_path (default '') directory in which network exists
--height (number)
--width (number)
--chain (string)
--units (string)
]]

local network_type = opt.type
local network = opt.network
local weights_filename = opt.weights
local network_directiory = opt.networkDirectory
local filename = paths.concat(opt.save, 'max_activations.hdf5')

-- GoogleNet (ImageNet) : 174, GoogLeNet(Cifar10) : 220, AlexNet(Cifar): 22
local chain = opt.chain
local units = _.map((opt.units):split(','),function(i,u)return tonumber(u) end)

-- PARAMS:
-- TODO: Have a set of parameters available to choose from in UI
-- Perhaps bestGoogLeNet,bestAlexNet, and bestLeNet
local reg_params = nnhelpers.bestAlexNet()
local max_iter = 400

cutorch.setDevice(1)

-- Load Network
local net = nnhelpers.loadNetwork(network_directiory,network,weights_filename,network_type)
local model = net.model
local size = net.croplen

local push_layer = nnhelpers.findLayerFromChain(model,chain)

-- model = nnhelpers.loadModel('./AlexNet(Cifar)_Color', 'model', './AlexNet(Cifar)_Color/snapshot_13_Model.t7', 'cuda')
-- model = nnhelpers.loadModel('./AlexNet(Cifar)_Grey', 'model', './AlexNet(Cifar)_Grey/snapshot_30_Model.t7', 'cuda')
-- model = nnhelpers.loadModel('./GoogLeNet(Cifar)', 'model', './GoogLeNet(Cifar)/old/snapshot_60_Model.t7', 'cuda')
-- model = torch.load('/home/lzeerwanklyn/Desktop/gradientAscentPlaygroundTorch/GoogLeNet(Cifar)/snapshot_60_Model.t7'):cuda()

-- Run Optimization
local mm = nnhelpers.getNetworkUntilPushLayer(model, push_layer):cuda()

local input_layer = mm:get(1)

input_layer.gradInput = torch.Tensor():cuda()

-- Adjust Input Image to match # channels, and image size:
local channels = input_layer.nInputPlane
if not channels then
  -- If first layer is a constant, then get # channels from second layer
  channels = mm:get(2).nInputPlane
end


local height  = opt.height
local width  = opt.width

local input_size = {1,channels,height,width}


local mean_image = torch.ones(image.lena():size())*0.5

if channels == 1 then
  mean_image = 0.333 * torch.add(torch.add(mean_image[1],mean_image[2]),mean_image[3])
end


local inputs = torch.Tensor(torch.LongStorage(input_size))
local im = image.scale(mean_image,input_size[3],input_size[4])
inputs[1] = im

-- local w = image.display{image=im, min=0, max=1}

-- If no units then , solve for all units in layer:
if units[1] == -1 then
  local outputs = nnhelpers.getNumOutputs(mm,channels,height,width)
  units = {}
  for i=1,outputs do
    units[i] = i
  end
end

-- Run Optimization:
g = nnhelpers.gaussianKernel(reg_params.blur_radius)

for i,push_unit in ipairs(units) do
  local mean = inputs:clone():cuda()
  local xx   = inputs:clone():cuda()
  local diffs = nnhelpers.generatePointGradient(mm,xx,push_unit,reg_params.push_spatial):cuda()

  for ii=1,max_iter do

    -- Run Forward with new image:
    xx = torch.cmin(torch.cmax(xx, 0),1)
    mm:forward(xx)

    -- Run backward:
    local gradInput = mm:backward(xx,diffs)

    -- Apply Regularizations:
    xx = nnhelpers.regularize(xx,gradInput,ii,g,reg_params)
    -- image.display{image=xx, win=w}

  end

  logmessage.display(0, 'Processed ' .. i .. '/' .. #units .. ' units')

  xx = (255*torch.cmin(torch.cmax(xx, 0),1)):int()
  local max_db = hdf5.open(filename, 'a')
  max_db:write(chain .. "/" .. push_unit-1 .. "/cropped", xx[1] )
  max_db:close()
end
