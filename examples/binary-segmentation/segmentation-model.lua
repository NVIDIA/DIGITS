-- return function that returns network definition
return function(params)
    -- get number of classes from external parameters
    local nclasses = params.nclasses or 1

    -- get number of channels from external parameters
    local channels = 1
    -- params.inputShape may be nil during visualization
    if params.inputShape then
        channels = params.inputShape[1]
    end

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

    -- --

    local net = nn.Sequential()

    -- conv1: 32 filters, 3x3 kernels, 1x1 stride, 1x1 pad
    net:add(backend.SpatialConvolution(channels,32,3,3,1,1,1,1)) -- C*H*W -> 32*H*W
    net:add(backend.ReLU())

    -- conv2: 1024 filters, 16x16 kernels, 16x16 stride, 0x0 pad
    -- on 16x16 inputs this is equivalent to a fully-connected layer with 1024 outputs
    net:add(backend.SpatialConvolution(32,1024,16,16,16,16,0,0)) -- 32*H*W -> 1024*H/16*W/16
    net:add(backend.ReLU())

    -- deconv: 1 filter, 16x16 kernel, 16x16 stride, 0x0 pad
    net:add(backend.SpatialFullConvolution(1024,1,16,16,16,16,0,0)) -- 1024*H/16*W/16 -> 1xH*W

    return {
        model = net,
        --loss = nn.MSECriterion(),
        loss = nn.SmoothL1Criterion(),
        --loss = nn.AbsCriterion(),
        trainBatchSize = 4,
        validationBatchSize = 32,
    }
end
