-- source: https://github.com/soumith/imagenet-multiGPU.torch/blob/master/models/alexnet_cudnn.lua

require 'nn'
if pcall(function() require('cudnn') end) then
   print('Using CuDNN backend')
   backend = cudnn
   convLayer = cudnn.SpatialConvolution
else
   print('Failed to load cudnn backend (is libcudnn.so in your library path?)')
   if pcall(function() require('cunn') end) then
       print('Falling back to legacy cunn backend')
   else
       print('Failed to load cunn backend (is CUDA installed?)')
       print('Falling back to legacy nn backend')
   end
   backend = nn -- works with cunn or nn
   convLayer = nn.SpatialConvolutionMM
end

local function inception(input_size, config)
   local concat = nn.Concat(2)
   if config[1][1] ~= 0 then
      local conv1 = nn.Sequential()
      conv1:add(convLayer(input_size, config[1][1],1,1,1,1)):add(backend.ReLU(true))
      concat:add(conv1)
   end

   local conv3 = nn.Sequential()
   conv3:add(convLayer(  input_size, config[2][1],1,1,1,1)):add(backend.ReLU(true))
   conv3:add(convLayer(config[2][1], config[2][2],3,3,1,1,1,1)):add(backend.ReLU(true))
   concat:add(conv3)

   local conv3xx = nn.Sequential()
   conv3xx:add(convLayer(  input_size, config[3][1],1,1,1,1)):add(backend.ReLU(true))
   conv3xx:add(convLayer(config[3][1], config[3][2],3,3,1,1,1,1)):add(backend.ReLU(true))
   conv3xx:add(convLayer(config[3][2], config[3][2],3,3,1,1,1,1)):add(backend.ReLU(true))
   concat:add(conv3xx)

   local pool = nn.Sequential()
   pool:add(nn.SpatialZeroPadding(1,1,1,1)) -- remove after getting cudnn R2 into fbcode
   if config[4][1] == 'max' then
      pool:add(backend.SpatialMaxPooling(3,3,1,1):ceil())
   elseif config[4][1] == 'avg' then
      local l = backend.SpatialAveragePooling(3,3,1,1)
      if backend == cudnn then l = l:ceil() end
      pool:add(l)
   else
      error('Unknown pooling')
   end
   if config[4][2] ~= 0 then
      pool:add(convLayer(input_size, config[4][2],1,1,1,1)):add(backend.ReLU(true))
   end
   concat:add(pool)

   return concat
end

function createModel(nChannels, nClasses)
   -- batch normalization added on top of convolutional layers in feature branch
   -- in order to help the network learn faster
   local features = nn.Sequential()
   features:add(nn.MulConstant(0.02))
   features:add(convLayer(nChannels,64,7,7,2,2,3,3)):add(backend.SpatialBatchNormalization(64,1e-3)):add(backend.ReLU(true))
   features:add(backend.SpatialMaxPooling(3,3,2,2):ceil())
   features:add(convLayer(64,64,1,1)):add(backend.SpatialBatchNormalization(64,1e-3)):add(backend.ReLU(true))
   features:add(convLayer(64,192,3,3,1,1,1,1)):add(backend.SpatialBatchNormalization(192,1e-3)):add(backend.ReLU(true))
   features:add(backend.SpatialMaxPooling(3,3,2,2):ceil())
   features:add(inception( 192, {{ 64},{ 64, 64},{ 64, 96},{'avg', 32}})) -- 3(a)
   features:add(inception( 256, {{ 64},{ 64, 96},{ 64, 96},{'avg', 64}})) -- 3(b)
   features:add(inception( 320, {{  0},{128,160},{ 64, 96},{'max',  0}})) -- 3(c)
   features:add(convLayer(576,576,2,2,2,2)):add(backend.SpatialBatchNormalization(576,1e-3))
   features:add(inception( 576, {{224},{ 64, 96},{ 96,128},{'avg',128}})) -- 4(a)
   features:add(inception( 576, {{192},{ 96,128},{ 96,128},{'avg',128}})) -- 4(b)
   features:add(inception( 576, {{160},{128,160},{128,160},{'avg', 96}})) -- 4(c)
   features:add(inception( 576, {{ 96},{128,192},{160,192},{'avg', 96}})) -- 4(d)

   local main_branch = nn.Sequential()
   main_branch:add(inception( 576, {{  0},{128,192},{192,256},{'max',  0}})) -- 4(e)
   main_branch:add(convLayer(1024,1024,2,2,2,2)):add(backend.SpatialBatchNormalization(1024,1e-3))
   main_branch:add(inception(1024, {{352},{192,320},{160,224},{'avg',128}})) -- 5(a)
   main_branch:add(inception(1024, {{352},{192,320},{192,224},{'max',128}})) -- 5(b)
   main_branch:add(backend.SpatialAveragePooling(7,7,1,1))
   main_branch:add(nn.View(1024):setNumInputDims(3))
   main_branch:add(nn.Linear(1024,nClasses))
   main_branch:add(nn.LogSoftMax())

   -- add auxiliary classifier here (thanks to Christian Szegedy for the details)
   local aux_classifier = nn.Sequential()
   local l = backend.SpatialAveragePooling(5,5,3,3)
   if backend == cudnn then l = l:ceil() end
   aux_classifier:add(l)
   aux_classifier:add(convLayer(576,128,1,1,1,1)):add(backend.SpatialBatchNormalization(128,1e-3))
   aux_classifier:add(nn.View(128*4*4):setNumInputDims(3))
   aux_classifier:add(nn.Linear(128*4*4,768))
   aux_classifier:add(backend.ReLU())
   aux_classifier:add(nn.Linear(768,nClasses))
   aux_classifier:add(backend.LogSoftMax())

   local splitter = nn.Concat(2)
   splitter:add(main_branch):add(aux_classifier)
   --local googlenet = nn.Sequential():add(features):add(splitter)

   local googlenet = nn.Sequential():add(features):add(main_branch)

   return googlenet
end

-- return function that returns network definition
return function(params)
    -- get number of classes from external parameters
    local nclasses = params.nclasses or 1
    -- adjust to number of channels in input images
    local channels = 1
    -- params.inputShape may be nil during visualization
    if params.inputShape then
        channels = params.inputShape[1]
        assert(params.inputShape[2]==256 and params.inputShape[3]==256, 'Network expects 256x256 images')
    end
    return {
        model = createModel(channels, nclasses),
        croplen = 224,
        trainBatchSize = 32,
        validationBatchSize = 16,
    }
end

