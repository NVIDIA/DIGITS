-- source: https://github.com/soumith/imagenet-multiGPU.torch/blob/master/models/alexnet_cudnn.lua

require 'nn'
if pcall(function() require('cudnn') end) then
   print('Using CuDNN backend')
   backend = cudnn
   convLayer = cudnn.SpatialConvolution
   convLayerName = 'cudnn.SpatialConvolution'
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
   convLayerName = 'nn.SpatialConvolutionMM'
end

function createModel(nGPU, channels)
   assert(nGPU == 1 or nGPU == 2, '1-GPU or 2-GPU supported for AlexNet')
   local features
   if nGPU == 1 then
      features = nn.Concat(2)
   else
      features = nn.ModelParallel(2)
   end

   local fb1 = nn.Sequential() -- branch 1
   fb1:add(convLayer(channels,48,11,11,4,4,2,2))       -- 224 -> 55
   fb1:add(backend.ReLU(true))
   fb1:add(backend.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   fb1:add(convLayer(48,128,5,5,1,1,2,2))       --  27 -> 27
   fb1:add(backend.ReLU(true))
   fb1:add(backend.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   fb1:add(convLayer(128,192,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(backend.ReLU(true))
   fb1:add(convLayer(192,192,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(backend.ReLU(true))
   fb1:add(convLayer(192,128,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(backend.ReLU(true))
   fb1:add(backend.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6

   local fb2 = fb1:clone() -- branch 2
   for k,v in ipairs(fb2:findModules(convLayerName)) do
      v:reset() -- reset branch 2's weights
   end

   features:add(fb1)
   features:add(fb2)

   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256*6*6, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   classifier:add(nn.Linear(4096, 1000))
   classifier:add(nn.LogSoftMax())

   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(features):add(classifier)

   return model
end

-- return function that returns network definition
return function(params)
    assert(params.ngpus<=1, 'Model supports only one GPU')
    -- adjust to number of channels in input images
    local channels = 1
    -- params.inputShape may be nil during visualization
    if params.inputShape then
        channels = params.inputShape[1]
        assert(params.inputShape[2]==256 and params.inputShape[3]==256, 'Network expects 256x256 images')
    end
    return {
        model = createModel(1, channels),
        croplen = 224,
        trainBatchSize = 100,
        validationBatchSize = 100,
    }
end


