-- Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.

require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'trepl'
require 'cutorch'
require 'image'
require 'cudnn'
require 'lfs'

package.path = debug.getinfo(1,"S").source:match[[^@?(.*[\/])[^\/]-$]] .."?.lua;".. package.path

require 'logmessage'
require 'Optimizer'
----------------------------------------------------------------------

--print 'processing options'

opt = lapp[[
-m,--resizeMode         (default squash)      	 Resize mode (squash/crop/fill/half_crop) for the input test image, if it's dimensions differs from those of Train DB images.
-t,--threads            (default 8)              number of threads
-p,--type               (default cuda)           float or cuda
-d,--devid              (default 1)              device ID (if using CUDA)
-o,--load               (string)                 directory that contains trained model weights
-n,--network            (string)                 Pretrained Model to be loaded
-e,--epoch              (default -1)             weight file of the epoch to be loaded
-i,--image              (string)                 the value to this parameter depends on "testMany" parameter. If testMany is 'no' then this parameter specifies single image that needs to be classified or else this parameter specifies the location of file which contains paths of multiple images that needs to be classified. Provide full path, if the image (or) images file is in different directory.
-s,--mean               (string)                 train images mean (saved as .jpg file)
-y,--ccn2               (default no)             should be 'yes' if ccn2 is used in network. Default : false

--testMany		(default no)             If this option is 'yes', then "image" input parameter should specify the file with all the images to be tested
--testUntil             (default -1)             specifies how many images in the "image" file to be tested. This parameter is only valid when testMany is set to "yes"
--crop                  (default no)             If this option is 'yes', all the images are randomly cropped into square image. And croplength is provided as --croplen parameter
--croplen               (default 0)              crop length. This is required parameter when crop option is provided
--subtractMean          (default yes)            If yes, subtracts the mean from images
--labels                (default labels.txt)     file contains label definitions
--useMeanPixel          (default 'no')           by default pixel-wise subtraction is done using the full mean matrix. If this option is 'yes' then mean pixel will be used instead of mean matrix
--snapshotPrefix        (default '')             prefix of the weights/snapshots
--networkDirectory      (default '')             directory in which network exists
--pythonPrefix        	(default 'python')       python version
--allPredictions        (default no)       	 If 'yes', displays all the predictions of an image instead of formatted topN results
]]


torch.setnumthreads(opt.threads)
cutorch.setDevice(opt.devid)


local snapshot_prefix = ''

if opt.snapshotPrefix ~= '' then
    snapshot_prefix = opt.snapshotPrefix
else
    snapshot_prefix = opt.network
end

local utils = require 'utils'

local data = require 'data'

local class_labels = data.loadLabels(opt.labels)

local img_mean=data.loadMean(opt.mean, opt.useMeanPixel)

local crop = opt.crop

local req_x = nil
local req_y = nil

-- if subtraction has to be done using the full mean matrix instead of mean pixel, then image cropping is possible only after subtracting the image from full mean matrix. In other cases, we can crop the image before subtracting mean.

if (opt.useMeanPixel == 'yes' or opt.subtractMean == 'no') and crop == 'yes' then
  req_x = opt.croplen
  req_y = opt.croplen
  crop = 'no'        -- as resize_image.py will take care of cropping as well
else
  req_x = img_mean["width"]
  req_y = img_mean["height"]
end

local cropX = nil
local cropY = nil
if crop == 'yes' then
  cropX = math.floor((img_mean["height"] - opt.croplen)/2) + 1
  cropY = math.floor((img_mean["width"] - opt.croplen)/2) + 1
end


-- If epoch for the trained model is not provided then select the latest trained model.
if opt.epoch == -1 then
  dir_name = paths.concat(opt.load)
  for file in lfs.dir(dir_name) do
    file_name = paths.concat(dir_name,file)
    if lfs.attributes(file_name,"mode") == "file" then
      if string.match(file, snapshot_prefix .. '_.*_Weights[.]t7') then
        parts=string.split(file,"_")
        value = tonumber(parts[#parts-1])
        if (opt.epoch < value) then
          opt.epoch = value
        end
      end
    end
  end
end

if opt.epoch == -1 then
    logmessage.display(2,'There are no pretrained model weights to test in this directory - ' .. paths.concat(opt.networkDirectory))
    return
end

package.path = paths.concat(opt.networkDirectory, "?.lua") ..";".. package.path

logmessage.display(0,'Loading network definition from ' .. paths.concat(opt.networkDirectory, opt.network))
local model = require (opt.network)
local using_ccn2 = opt.ccn2

-- if ccn2 is used in network, then set using_ccn2 value as 'yes'
if ccn2 ~= nil then
  using_ccn2 = 'yes'
end

local weights, gradients = model:getParameters()

logmessage.display(0, 'Loading ' .. paths.concat(opt.load, snapshot_prefix .. '_' .. opt.epoch .. '_Weights.t7') .. ' file')

local weights_filename = paths.concat(opt.load, snapshot_prefix .. '_' .. opt.epoch .. '_Weights.t7')
weights:copy(torch.load(weights_filename))

if opt.type =='cuda' then
  model:cuda()
end

-- as we want to classify, let's disable dropouts by enabling evaluation mode
model:evaluate()

local function preprocess(img_path)

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

    local im = image.load(img_path)

    -- resize image to match with the required size. Required size may be mean file size or crop size input
    if (img_mean["channels"] ~= im:size(1)) or (req_y ~= im:size(2)) or (req_x ~= im:size(3)) then
        im = utils.resizeImage(im, req_y, req_x, img_mean["channels"],opt.resizeMode)
    end
    -- Torch image.load() always loads image with each pixel value between 0-1. As during training, images were taken from LMDB directly, their pixel values ranges from 0-255. As, model was trained with images whose pixel values are between 0-255, we may have to convert test image also to have 0-255 for each pixel.
    im=im*255

    -- Depending on the function arguments, image preprocess may include conversion from RGB to BGR and mean subtraction, image resize after mean subtraction
    local image_preprocessed = data.PreProcess(im, img_mean["mean"], opt.subtractMean, img_mean["channels"], 'no', crop, false, cropX, cropY, opt.croplen)
    return image_preprocessed
end


local inputs = nil
local batch_size = 0
local predictions = nil
local topN = 5    -- displays top 5 predictions
if topN > #class_labels then
    topN = #class_labels
end

local val,classes = nil,nil
local counter = 0
local index = 0

-- if ccn2 is used, then batch size of the input should be atleast 32
if using_ccn2 == 'yes' or opt.testMany == 'yes' then
  batch_size = 32
else
  batch_size = 1
end

if opt.crop == 'yes' then   -- notice that here "opt.crop" is used, instead of "crop", as there are a chances that "crop" variable is getting overriden in the above instructions
  inputs = torch.Tensor(batch_size, img_mean["channels"], opt.croplen, opt.croplen)
else
  inputs = torch.Tensor(batch_size, img_mean["channels"], img_mean["height"], img_mean["width"])
end

-- predict batch and display the topN predictions for the images in batch
local function predictBatch(inputs)
  if opt.type == 'float' then
    predictions = model:forward(inputs:float())
  elseif opt.type =='cuda' then
    predictions = model:forward(inputs:cuda())
  end
  -- sort the outputs of SoftMax layer in decreasing order
  for i=1,counter do
    index = index + 1
    if opt.allPredictions == 'no' then
      --display topN predictions of each image
      val,classes = predictions[{i,{}}]:float():sort(true)
      for j=1,topN do
        -- output format : LABEL_ID (LABEL_NAME) CONFIDENCE
        logmessage.display(0,'For image ' .. index ..', predicted class '..tostring(j)..': ' .. classes[j] .. ' (' .. class_labels[classes[j]] .. ') ' .. math.exp(val[j]))
      end
    else
      val = predictions[{i,{}}]:float()
      allPredictions = ''
      for j=1,val:size(1) do
        allPredictions = allPredictions .. ' ' .. math.exp(val[j])
      end
      logmessage.display(0,'Predictions for image ' .. index ..': '..allPredictions)
    end
  end
end

if opt.testMany == 'yes' then
  local file = io.open(opt.image)
  if file then

    for line in file:lines() do
      counter = counter + 1
      local image_path = line:match( "^%s*(.-)%s*$" )
      inputs[counter] = preprocess(image_path)

      if counter == batch_size then
        predictBatch(inputs)
        counter = 0
      end
      if (index+counter) == opt.testUntil then                   -- Here, index+counter represents total number of images read from file
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
        predictBatch(inputs)

      else
        predictBatch(inputs:narrow(1,1,counter))
      end
    end
  else
    logmessage.display(2,'Image file not found : ' .. opt.image)
  end

else
  -- only one image needs to be predicted
  inputs[1]=preprocess(opt.image)
  if using_ccn2 == 'yes' then
    for j=2,batch_size do
      inputs[j] = inputs[1]             -- replicate the first image in entire inputs tensor
    end
  end
  counter = 1      -- here counter is set, so that predictBatch() method displays only the predictions of first image
  predictBatch(inputs)
end

