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
-m,--mode               (default half_crop)      Resize mode (squash/crop/fill/half_crop) for the input test image, if it's dimensions differs from those of Train DB images.
-t,--threads            (default 8)              number of threads
-p,--type               (default cuda)           float or cuda
-d,--devid              (default 1)              device ID (if using CUDA)
-o,--load               (string)                 directory that contains trained model weights
-n,--network            (string)                 Pretrained Model to be loaded
-e,--epoch              (default -1)             weight file of the epoch to be loaded
-i,--image              (string)                 image that needs to be classified. Provide full path, if the image is in different directory.
-s,--mean               (string)                 train images mean (saved as .jpg file)
-y,--ccn2               (default no)             should be 'yes' if ccn2 is used in network. Default : false

--crop                  (default no)             If this option is 'yes', all the images are randomly cropped into square image. And croplength is provided as --croplen parameter
--croplen               (default 0)              crop length. This is required parameter when crop option is provided
--subtractMean          (default yes)            If yes, subtracts the mean from images
--labels                (default labels.txt)     file contains label definitions
--useMeanPixel          (default 'no')           by default pixel-wise subtraction is done using the full mean matrix. If this option is 'yes' then mean pixel will be used instead of mean matrix
--snapshotPrefix        (default '')             prefix of the weights/snapshots
--networkDirectory      (default '')             directory in which network exists
--pythonPrefix        	(default 'python')       python version
]]


torch.setnumthreads(opt.threads)
cutorch.setDevice(opt.devid)


local snapshot_prefix = ''

if opt.snapshotPrefix ~= '' then
    snapshot_prefix = opt.snapshotPrefix
else
    snapshot_prefix = opt.network
end

local data = require 'data'

local class_labels = data.loadLabels(opt.labels)


-- if image doesn't exists in path, check whether provided path is an URL, if URL download it else display error message and return. This function is useful only when the test code was run from commandline.
if not paths.filep(opt.image) then
  if (opt.image:find("^http[.]") ~= nil) or (opt.image:find("^https[.]") ~= nil) or (opt.image:find("^www[.]") ~= nil) then
    os.execute('wget '..opt.image)
    opt.image = opt.image:match( "([^/]+)$" )
  else
    logmessage.display(2,'Image not found : ' .. opt.image)
    return
  end
end

-- If epoch for the trained model is not provided then select the latest trained model.
if opt.epoch == -1 then
  dir_name = paths.concat(opt.load)
  for file in lfs.dir(dir_name) do
    file_name = paths.concat(dir_name,file)
    if lfs.attributes(file_name,"mode") == "file" then
      if string.match(file,'_.*_Weights.t7') then
        parts=string.split(file,"_")
        value = tonumber(parts[#parts-1])
        if (opt.epoch < value) then
          opt.epoch = value
        end
      end
    end
  end
end


local im = image.load(opt.image)

local img_mean=data.loadMean(opt.mean, opt.useMeanPixel)


local req_x = nil
local req_y = nil

if (opt.useMeanPixel == 'yes' or opt.subtractMean == 'no') and opt.crop == 'yes' then
  req_x = opt.croplen
  req_y = opt.croplen
  opt.crop = 'no'        -- as resize_image.py will take care of cropping as well
else
  req_x = img_mean["height"]
  req_y = img_mean["width"]
end


if (img_mean["channels"] ~= im:size(1)) or (req_x ~= im:size(2)) or (req_y ~= im:size(3)) then
   os.execute(opt.pythonPrefix .. ' ' .. paths.concat(debug.getinfo(1,"S").source:sub(2),"../../resize_image.py") .. ' ' .. opt.image .. ' ' .. opt.image .. ' ' .. req_x .. ' ' .. req_y .. ' -c ' .. img_mean["width"] .. ' -m ' .. opt.mode)
   im = image.load(opt.image)
end

if opt.type == 'float' then
    logmessage.display(0,'switching to floats')
    torch.setdefaulttensortype('torch.FloatTensor')

elseif opt.type =='cuda' then
    require 'cunn'
    logmessage.display(0,'switching to CUDA')
    --model:cuda()
    torch.setdefaulttensortype('torch.CudaTensor')
end


package.path =  paths.concat(opt.networkDirectory, "?.lua") ..";".. package.path

local model_filename = paths.concat(opt.networkDirectory, opt.network)
logmessage.display(0,'Loading Model: ' .. model_filename)
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

-- if the image size is same as required crop size image.

local cropX = nil
local cropY = nil
if opt.crop == 'yes' then
  cropX = math.floor((img_mean["height"] - opt.croplen)/2) + 1
  cropY = math.floor((img_mean["width"] - opt.croplen)/2) + 1
end

-- Torch image.load() always loads image with each pixel value between 0-1. As during training, images were taken from LMDB directly, their pixel values ranges from 0-255. As, model was trained with images whose pixel values are between 0-255, we may have to convert test image also to have 0-255 for each pixel.
im=im*255

-- Image preporcess including resize, conversion from RGB to BGR and mean subtraction, depending on input parameters
local image_preprocessed = data.PreProcess(im, img_mean["mean"], opt.subtractMean, img_mean["channels"], 'no', opt.crop, false, cropX, cropY, opt.croplen)

local inputs

 -- if ccn2 is used, then batch size of the input should be atleast 32
if using_ccn2 == 'yes' then
  inputs = torch.Tensor(32, image_preprocessed:size(1), image_preprocessed:size(2), image_preprocessed:size(3))

  for i=1,32 do
    inputs[i]=image_preprocessed
  end
else
  inputs = torch.Tensor(1, image_preprocessed:size(1), image_preprocessed:size(2), image_preprocessed:size(3))
  inputs[1]=image_preprocessed
end

if opt.type == 'float' then
  inputs=inputs:float()
elseif opt.type =='cuda' then
  inputs=inputs:cuda()
end

local y = model:forward(inputs)

-- for the outputs of SoftMax layer sort them in decreasing order
val,classes = y[{1,{}}]:float():sort(true)

-- output format : LABEL_ID (LABEL_NAME) CONFIDENCE
for i=1,5 do
  logmessage.display(0,'Predicted class '..tostring(i)..': ' .. classes[i] .. ' (' .. class_labels[classes[i]] .. ') ' .. math.exp(val[i]))
end



