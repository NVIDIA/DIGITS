require 'nn'
require 'cunn'
require 'inn'

-- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
-- -- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
--
local model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(1,20,5,5,1,1,0)) -- 1*28*28 -> 20*24*24
model:add(inn.SpatialMaxPooling(2, 2, 2, 2)) -- 20*24*24 -> 20*12*12
model:add(nn.SpatialConvolutionMM(20,50,5,5,1,1,0)) -- 20*12*12 -> 50*8*8
model:add(inn.SpatialMaxPooling(2,2,2,2)) --  50*8*8 -> 50*4*4
model:add(nn.View(-1):setNumInputDims(3))  -- 50*4*4 -> 800 
model:add(nn.Linear(800,500))  -- 800 -> 500
model:add(nn.ReLU())  
model:add(nn.Linear(500, 10))  -- 500 -> 10
model:add(nn.SoftMax())
model:cuda()
return model

