require 'nn'

-- -- This is a LeNet model. For more information: http://yann.lecun.com/exdb/lenet/

local lenet = nn.Sequential()
lenet:add(nn.MulConstant(0.00390625))
lenet:add(nn.SpatialConvolution(1,20,5,5,1,1,0)) -- 1*28*28 -> 20*24*24
lenet:add(nn.SpatialMaxPooling(2, 2, 2, 2)) -- 20*24*24 -> 20*12*12
lenet:add(nn.SpatialConvolution(20,50,5,5,1,1,0)) -- 20*12*12 -> 50*8*8
lenet:add(nn.SpatialMaxPooling(2,2,2,2)) --  50*8*8 -> 50*4*4
lenet:add(nn.View(-1):setNumInputDims(3))  -- 50*4*4 -> 800
lenet:add(nn.Linear(800,500))  -- 800 -> 500
lenet:add(nn.ReLU())
lenet:add(nn.Linear(500, 10))  -- 500 -> 10
lenet:add(nn.LogSoftMax())

-- return function that returns network definition
return function(params)
    assert(params.ngpus<=1, 'Model supports only CPU or single-GPU')
    return {
        model = lenet,
        loss = nn.ClassNLLCriterion()
    }
end


