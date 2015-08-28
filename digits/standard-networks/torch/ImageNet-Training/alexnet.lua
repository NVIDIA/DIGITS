-- Copyright (c) 2015 Elad Hoffer

require 'cudnn'
require 'cunn'
require 'ccn2'
   local SpatialConvolution = nn.SpatialConvolutionMM--lib[1]
   local SpatialMaxPooling = cudnn.SpatialMaxPooling--lib[2]
   local ReLU = nn.ReLU--lib[3]

   -- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
   -- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
   local features = nn.Sequential()
   features:add(SpatialConvolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
   features:add(ReLU())
   features:add(SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   features:add(SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
   features:add(ReLU())
   features:add(SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   features:add(SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
   features:add(ReLU())
   features:add(SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(ReLU())
   features:add(SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
   features:add(ReLU())
   features:add(SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

   local classifier = nn.Sequential()
   classifier:add(nn.View(256*7*7))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256*7*7, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Linear(4096, 20))
   classifier:add(nn.LogSoftMax())

   local model = nn.Sequential()
   model:add(features):add(classifier)

   return model

