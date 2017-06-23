-- return function that returns network definition
return function(params)
    assert(params.ngpus<=1, 'Model does not support multi-GPU training because of shared weights')

    local channels = 1
    -- params.inputShape may be nil during visualization
    if params.inputShape then
        channels = params.inputShape[1]
        assert(params.inputShape[2]==28 and params.inputShape[3]==28, 'Network expects 28x28 images')
    end

    -- adjust to number of channels in input images - default to 1 channel
    -- during model visualization
    local channels = (params.inputShape and params.inputShape[1]) or 1

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

    local lenet = nn.Sequential() -- expected input: Nx1x28x28
    lenet:add(nn.Reshape(1,28,28))
    lenet:add(nn.MulConstant(0.03))
    lenet:add(backend.SpatialConvolution(1,20,5,5,1,1,0)) -- 1*28*28 -> 20*24*24
    lenet:add(backend.SpatialMaxPooling(2, 2, 2, 2)) -- 20*24*24 -> 20*12*12
    lenet:add(backend.SpatialConvolution(20,50,5,5,1,1,0)) -- 20*12*12 -> 50*8*8
    lenet:add(backend.SpatialMaxPooling(2,2,2,2)) --  50*8*8 -> 50*4*4
    lenet:add(nn.View(-1):setNumInputDims(3))  -- 50*4*4 -> 800
    lenet:add(nn.Linear(800,500))  -- 800 -> 500
    lenet:add(backend.ReLU())
    lenet:add(nn.Linear(500, 2))  -- 500 -> 2 (reduce to two features for plotting)
    lenet:add(nn.Reshape(1,2))

    local parallel = nn.Parallel(2,2) -- split along channel dimension
    parallel:add(lenet) -- left branch
    parallel:add(lenet:clone('weight', 'bias', 'gradWeight', 'gradBias')) -- right branch, shared weights

    local siamese = nn.Sequential()
    siamese:add(nn.Narrow(2,2,2)) -- drop red channel
    siamese:add(parallel) -- add parallel features
    siamese:add(nn.SplitTable(2))

    local criterion = nn.CosineEmbeddingCriterion(0.8)

    function siameseLabelHook(input, dblabel)
        -- cosine embedding criterion requires negative samples to be
        -- assigned class -1
        dblabel[torch.eq(dblabel,0)]=-1
        return dblabel
    end

    return {
        model = siamese,
        loss = criterion,
        trainBatchSize = 8,
        validationBatchSize = 8,
        labelHook = siameseLabelHook
    }
end
