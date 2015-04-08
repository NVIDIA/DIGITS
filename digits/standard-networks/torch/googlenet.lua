require 'nn'
require 'cunn'
require 'cudnn'
require 'ccn2'
local opt = opt or {type = 'cuda', net='new'}
local DimConcat = 2

---------------------------------------Inception Modules-------------------------------------------------
local Inception = function(nInput, n1x1, n3x3r, n3x3, n5x5r, n5x5, nPoolProj)
    local InceptionModule = nn.DepthConcat(DimConcat)
    InceptionModule:add(nn.Sequential():add(nn.SpatialConvolutionMM(nInput,n1x1,1,1)))
    InceptionModule:add(nn.Sequential():add(nn.SpatialConvolutionMM(nInput,n3x3r,1,1)):add(nn.ReLU()):add(nn.SpatialConvolutionMM(n3x3r,n3x3,3,3,1,1,1)))
    InceptionModule:add(nn.Sequential():add(nn.SpatialConvolutionMM(nInput,n5x5r,1,1)):add(nn.ReLU()):add(nn.SpatialConvolutionMM(n5x5r,n5x5,5,5,1,1,2)))
    InceptionModule:add(nn.Sequential():add(cudnn.SpatialMaxPooling(3,3,1,1)):add(nn.SpatialConvolutionMM(nInput,nPoolProj,1,1)))
    return InceptionModule
end

local AuxileryClassifier = function(nInput)
    local C = nn.Sequential()
    C:add(cudnn.SpatialAveragePooling(5,5,3,3))
    C:add(nn.SpatialConvolutionMM(nInput,128,1,1))
    C:add(nn.ReLU())
    C:add(nn.Reshape(128*4*4))
    C:add(nn.Linear(128*4*4,1024))
    C:add(nn.Dropout(0.7))
    C:add(nn.Linear(1024,1000))
    C:add(nn.LogSoftMax())
    return C
end
-----------------------------------------------------------------------------------------------------------

local Net = nn.Sequential()

local SubNet1 = nn.Sequential()
SubNet1:add(nn.SpatialConvolutionMM(3,64,7,7,2,2,4))
SubNet1:add(nn.ReLU())
SubNet1:add(cudnn.SpatialMaxPooling(3,3,2,2))
--SubNet1:add(ccn2.SpatialResponseNormalization(3))
SubNet1:add(nn.SpatialConvolutionMM(64,64,1,1))
SubNet1:add(nn.ReLU())
SubNet1:add(nn.SpatialConvolutionMM(64,192,3,3,1,1,1))
SubNet1:add(nn.ReLU())
--SubNet1:add(ccn2.SpatialResponseNormalization(3))
SubNet1:add(nn.SpatialZeroPadding(1,1,1,1))
SubNet1:add(cudnn.SpatialMaxPooling(3,3,2,2))



SubNet1:add(Inception(192,64,96,128,16,32,32))
SubNet1:add(nn.ReLU())
SubNet1:add(Inception(256,128,128,192,32,96,64))
SubNet1:add(nn.ReLU())
SubNet1:add(nn.SpatialZeroPadding(1,1,1,1))
SubNet1:add(cudnn.SpatialMaxPooling(3,3,2,2))
SubNet1:add(Inception(480,192,96,208,16,48,64))
SubNet1:add(nn.ReLU())



local SubNet2 = nn.Sequential()
SubNet2:add(SubNet1)
SubNet2:add(Inception(512,160,112,224,24,64,64))
SubNet2:add(nn.ReLU())
SubNet2:add(Inception(512,128,128,256,24,64,64))
SubNet2:add(nn.ReLU())
SubNet2:add(Inception(512,112,144,288,32,64,64))
SubNet2:add(nn.ReLU())



Net:add(SubNet2)
Net:add(Inception(528,256,160,320,32,128,128))
Net:add(nn.ReLU())
Net:add(nn.SpatialZeroPadding(1,1,1,1))
Net:add(cudnn.SpatialMaxPooling(3,3,2,2))


Net:add(Inception(832,256,160,320,32,128,128))
Net:add(nn.ReLU())
Net:add(Inception(832,384,192,384,48,128,128))
Net:add(nn.ReLU())
Net:add(cudnn.SpatialAveragePooling(7,7,1,1))
Net:add(nn.Dropout(0.4))
Net:add(nn.Reshape(1024))
Net:add(nn.Linear(1024,1000))
Net:add(nn.LogSoftMax())

local Classifier0 = nn.Sequential()
Classifier0:add(SubNet1)
Classifier0:add(AuxileryClassifier(512))

local Classifier1 = nn.Sequential()
Classifier1:add(SubNet2)
Classifier1:add(AuxileryClassifier(528))

--
--Net:cuda()
------ Loss: NLL
--Net = Classifier0
local loss = nn.ClassNLLCriterion()
----------------------------------------------------------------------
if opt.type == 'cuda' then
    Net:cuda()
    loss:cuda()
end

----------------------------------------------------------------------
print '==> flattening Net parameters'

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
--end

local w,dE_dw = Net:getParameters()

local t = torch.load('Weights')
w:copy(t)
--
--local t = torch.tic(); y = Net:forward(torch.rand(128,3,224,224):cuda()) ; cutorch.synchronize(); print(torch.tic()-t)
--print(SubNet1.modules[9].output:size())
--print(y:size())


-- return package:
return {
    Net = Net,
    Weights = w,
    Grads = dE_dw,
    Loss = loss
}

