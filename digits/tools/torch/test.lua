-- Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.

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
----------------------------------------------------------------------

--print 'processing options'

opt = lapp[[
-t,--threads (default 8) number of threads
-p,--type (default cuda) float or cuda
-d,--devid (default 1) device ID (if using CUDA)
-n,--network (string) Pretrained Model to be loaded
-i,--image (string) the value to this parameter depends on "testMany" parameter. If testMany is 'no' then this parameter specifies single image that needs to be classified or else this parameter specifies the location of file which contains paths of multiple images that needs to be classified. Provide full path, if the image (or) images file is in different directory.
-s,--mean (default '') train images mean (saved as .jpg file)
-y,--ccn2 (default no) should be 'yes' if ccn2 is used in network. Default : false
-s,--save (default .) save directory

--testMany (default no) If this option is 'yes', then "image" input parameter should specify the file with all the images to be tested
--testUntil (default -1) specifies how many images in the "image" file to be tested. This parameter is only valid when testMany is set to "yes"
--subtractMean (default 'image') Select mean subtraction method. Possible values are 'image', 'pixel' or 'none'.
--labels (default '') file contains label definitions
--snapshot (string) Path to snapshot to load
--networkDirectory (default '') directory in which network exists
--allPredictions (default no) If 'yes', displays all the predictions of an image instead of formatted topN results
--visualization (default no) Create HDF5 database with layers weights and activations. Depends on --testMany~=yes
--crop (default no) If this option is 'yes', all the images are cropped into square image. And croplength is provided as --croplen parameter
--croplen (default 0) crop length. This is required parameter when crop option is provided
]]


torch.setnumthreads(opt.threads)

if pcall(function() require('cutorch') end) then
    require 'cunn'
    if opt.type =='cuda' then
        cutorch.setDevice(opt.devid)
    end
end

if opt.testMany=='yes' and opt.visualization=='yes' then
    logmessage.display(1,'testMany==yes => disabling visualizations')
    opt.visualization = 'no'
end

if opt.crop == 'yes' then
    assert(opt.croplen > 0, "Crop length should be specified if crop is 'yes'")
else
    opt.croplen = nil
end

local utils = require 'utils'

local data = require 'data'

local class_labels
if opt.labels ~= '' then
    class_labels = data.loadLabels(opt.labels)
else
    assert(opt.allPredictions == 'yes', 'Regression problems must return all predictions')
end

local meanTensor
if opt.subtractMean ~= 'none' then
    assert(opt.mean ~= '', 'subtractMean parameter not set to "none" yet mean image path is unset')
    logmessage.display(0,'Loading mean tensor from '.. opt.mean ..' file')
    meanTensor = data.loadMean(opt.mean, opt.subtractMean == 'pixel')
end

-- returns shape of input tensor (adjusting to desired crop length if specified)
function getInputTensorShape(img, optCropLen)
    -- create new tensor containing dimensions of input image
    local inputShape = torch.Tensor(torch.Storage(img:size():size()):copy(img:size()))
    if optCropLen then
        -- crop length was specified on command line, overwrite dimensions of input image
        inputShape[2] = optCropLen
        inputShape[3] = optCropLen
    end
    return inputShape
end

-- loads network model and performs various operations:
-- * reload weights from snapshot
-- * move model to CUDA or FLOAT
-- * adjust last layer to match number of labels (for classification networks)
function loadNetwork(dir, name, labels, weightsFile, tensorType, inputTensorShape)
    package.path = paths.concat(dir, "?.lua") ..";".. package.path

    logmessage.display(0,'Loading network definition from ' .. paths.concat(dir, name))
    local parameters = {
        ngpus = (tensorType =='cuda') and 1 or 0,
        nclasses = (labels ~= nil) and #labels or nil,
        inputShape = inputTensorShape,
    }
    if nn.DataParallelTable then
        -- set number of GPUs to use when deserializing model
        nn.DataParallelTable.deserializeNGPUs = parameters.ngpus
    end
    local network = require (name)(parameters)

    local model
    if (string.find(weightsFile, '_Weights')) then
        -- snapshot was created by dumping learnable weights/bias into a file
        -- this means other parameters such as those used for batch normalization
        -- could be missing, which could lead to incorrect predictions
        model = network.model

        -- allow user to fine tune model (we need to do this *before* loading weights)
        if network.fineTuneHook then
            logmessage.display(0,'Calling user-defined fine tuning hook...')
            model = network.fineTuneHook(model)
        end

        -- load parameters from snapshot
        local weights, gradients = model:getParameters()

        logmessage.display(0, 'Loading ' .. weightsFile)
        local savedWeights = torch.load(weightsFile)
        weights:copy(savedWeights)
    else
        -- the full model was saved
        assert(string.find(weightsFile, '_Model'))
        model = torch.load(weightsFile)
        network.model = model
    end

    if tensorType =='cuda' then
        model:cuda()
    else
        model:float()
    end

    -- as we want to classify, let's disable dropouts by enabling evaluation mode
    model:evaluate()

    return network
end

local using_ccn2 = opt.ccn2

local topN
if class_labels then
    topN = 5 -- displays top 5 predictions
    if topN > #class_labels then
        topN = #class_labels
    end
end

-- if ccn2 is used in network, then set using_ccn2 value as 'yes'
if ccn2 ~= nil then
    using_ccn2 = 'yes'
end

local weights_filename = opt.snapshot

-- loads an image from specified path (file system or URL)
local function loadImage(img_path)
    -- if image doesn't exists in path, check whether provided path is an URL, if URL download it else display error message and return. This function is useful only when the test code was run from commandline.
    if not paths.filep(img_path) then
        if (img_path:find("^http[.]") ~= nil) or (img_path:find("^https[.]") ~= nil) or (img_path:find("^www[.]") ~= nil) then
            os.execute('wget '..img_path)
            img_path = img_path:match( "([^/]+)$" )
        else
            logmessage.display(2,'Image not found : ' .. img_path)
            return nil
        end
    end

    local im = image.load(img_path):type('torch.FloatTensor'):contiguous()

    -- Torch image.load() always loads image with each pixel value between 0-1. As during training, images were taken from LMDB directly, their pixel values ranges from 0-255. As, model was trained with images whose pixel values are between 0-255, we may have to convert test image also to have 0-255 for each pixel.
    im:mul(255)

    return im
end

-- preprocess image (subtract mean and crop)
local function preprocess(im, mean, croplen)

    -- Depending on the function arguments, image preprocess may include conversion from RGB to BGR and mean subtraction, image resize after mean subtraction
    local image_preprocessed = data.PreProcess(im,    -- input image
                                               false, -- test mode
                                               mean,  -- mean
                                               {}     -- augOpt
                                               )

    -- crop to match network expected input dimensions
    if croplen then
        image_size = image_preprocessed:size()
        assert(image_size[2] >= croplen and image_size[3] >= croplen, "Image must be larger than crop length in all spatial dimensions")
        c = (image_size[2]-croplen)/2 + 1
        image_preprocessed = data.PreProcess(image_preprocessed, -- input image
                                             false, --test mode
                                             nil,   -- no mean subtraction (this was done before)
                                             {crop={use=true,X=c,Y=c,len=croplen}}
                                             )
    end
    return image_preprocessed
end

local batch_size = 0
local predictions = nil

local val,classes = nil,nil
local counter = 0
local index = 0

-- if ccn2 is used, then batch size of the input should be atleast 32
if using_ccn2 == 'yes' or opt.testMany == 'yes' then
    batch_size = 32
else
    batch_size = 1
end

-- tensor of inputs batch size * channels * height * width
local inputs

-- predict batch and display the topN predictions for the images in batch
local function predictBatch(inputs, model)
    if opt.type == 'float' then
        predictions = model:forward(inputs:float())
    elseif opt.type =='cuda' then
        predictions = model:forward(inputs:cuda())
    end

    for i=1,counter do
        local prediction
        index = index + 1

        if type(predictions) == 'table' then
            prediction = {}
            for j=1,#predictions do
                prediction[j] = predictions[j][i]
            end
        else
            assert(torch.isTensor(predictions))
            if predictions:nDimension() == 1 then
                -- some networks drop the batch dimension when fed with only one training example
                assert(counter == 1, "Expect only one sample when prediction has dimensionality of 1 - counter=" .. counter)
                prediction = predictions
            else
                prediction = predictions[i]
            end
        end

        if class_labels then
            -- assume logSoftMax and convert to probabilities
            prediction:exp()
        end

        if opt.allPredictions == 'no' then
            --display topN predictions of each image
            prediction,classes = prediction:float():sort(true)
            for j=1,topN do
                -- output format : LABEL_ID (LABEL_NAME) CONFIDENCE
                logmessage.display(0,'For image ' .. index ..', predicted class '..tostring(j)..': ' .. classes[j] .. ' (' .. class_labels[classes[j]] .. ') ' .. prediction[j])
            end
        else
            allPredictions = utils.dataToJson(prediction)
            logmessage.display(0,'Predictions for image ' .. index ..': '..allPredictions)
        end
    end
end

if opt.testMany == 'yes' then
    local network
    local model
    local file = io.open(opt.image)
    if file then
        for line in file:lines() do
            counter = counter + 1
            local image_path = line:match( "^%s*(.-)%s*$" )
            local im = loadImage(image_path)
            local inputShape = getInputTensorShape(im, opt.croplen)
            if not network then
                -- load model now - we need to wait after we have read at least one image to be able to
                -- determine the shape of the input tensor and provide it to the user-defined function
                network = loadNetwork(opt.networkDirectory, opt.network, class_labels, weights_filename, opt.type, inputShape)
                model = network.model
            end
            local input = preprocess(im, meanTensor, opt.croplen or network.croplen)
            assert(input ~= nil, "Failed to load image")
            if not inputs then
                inputs = torch.Tensor(batch_size, input:size(1), input:size(2), input:size(3))
            end
            inputs[counter] = input

            if counter == batch_size then
                predictBatch(inputs, model)
                counter = 0
            end
            if (index+counter) == opt.testUntil then -- Here, index+counter represents total number of images read from file
                break
            end

        end
        -- still some images needs to be predicted.
        if counter > 0 then
            -- if ccn2 is used, then batch size of the input should be atleast 32. So, append additional images at the end to make the same as batch size (which is 32)
            if using_ccn2 == 'yes' then
                for j=counter+1,batch_size do
                    inputs[j] = inputs[counter]
                end
                predictBatch(inputs, model)

            else
                predictBatch(inputs:narrow(1,1,counter), model)
            end
        end
    else
        logmessage.display(2,'Image file not found : ' .. opt.image)
    end

else
    -- only one image needs to be predicted
    local im = loadImage(opt.image)
    local inputShape = getInputTensorShape(im, opt.croplen)
    local network = loadNetwork(opt.networkDirectory, opt.network, class_labels, weights_filename, opt.type, inputShape)
    local model = network.model
    local input = preprocess(im, meanTensor, opt.croplen or network.croplen)
    assert(input ~= nil, "Failed to load image")
    inputs = torch.Tensor(1, input:size(1), input:size(2), input:size(3))
    inputs[1] = input
    if using_ccn2 == 'yes' then
        for j=2,batch_size do
            inputs[j] = inputs[1] -- replicate the first image in entire inputs tensor
        end
    end
    counter = 1 -- here counter is set, so that predictBatch() method displays only the predictions of first image
    predictBatch(inputs, model)
    if opt.visualization=='yes' then
        local filename = paths.concat(opt.save, 'vis.h5')
        logmessage.display(0,'Saving visualization to ' .. filename)
        local vis_db = hdf5.open(filename, 'w')
        local layer_id = 1
        for i,layer in ipairs(model:listModules()) do
            local activations = layer.output
            local weights = layer.weight
            local bias = layer.bias
            name = tostring(layer)
            -- convert 'name' string to Tensor as torch.hdf5 only
            -- accepts Tensor objects
            tname = utils.stringToTensor(name)
            if weights ~= nil then
                vis_db:write('/layers/'..layer_id..'/weights', weights:float())
            end
            if bias ~= nil then
                vis_db:write('/layers/'..layer_id..'/bias', bias:float())
            end
            if type(activations) == 'table' then
                for k=1,#activations do
                    name_activation = name..'.'..k
                    tname_activation = utils.stringToTensor(name_activation)
                    vis_db:write('/layers/'..layer_id..'/name', tname_activation )
                    vis_db:write('/layers/'..layer_id..'/activations', activations[k]:float())
                    layer_id = layer_id + 1
                end
            else
                vis_db:write('/layers/'..layer_id..'/name', tname )
                vis_db:write('/layers/'..layer_id..'/activations', activations:float())
                layer_id = layer_id + 1
            end
        end
        vis_db:close()
    end
end

