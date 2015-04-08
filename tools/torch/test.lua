require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'Optimizer'
require 'trepl'
require 'cutorch'
require 'image'
require 'ccn2'
require 'cunn'
require 'cudnn'
require 'lfs'
----------------------------------------------------------------------

--print '==> processing options'

opt = lapp[[
-m,--mode               (default half_crop)      Resize mode (squash/crop/fill/half_crop)
-t,--threads            (default 8)              number of threads
-p,--type               (default cuda)           float or cuda
-d,--devid              (default 1)              device ID (if using CUDA)
-o,--load               (string)                 directory that contains trained model weights
-n,--network            (string)                 Model - must return valid network. Available - {CaffeRef_Model, AlexNet_Model, NiN_Model, OverFeat_Model}
-e,--epoch              (default -1)             weight file of the epoch to be loaded
-i,--image              (string)                 image that needs to be classified
-s,--mean               (string)                 train images mean (saved as .jpg file)
-y,--ccn2               (default no)             should be 'yes' if ccn2 is used in network. Default : false
-v,--visualize          (default 0)              visualizing results
]]


torch.setnumthreads(opt.threads)
cutorch.setDevice(opt.devid)

-- Helper functions

-- Loads the mapping from net outputs to human readable labels
function load_classes()
  local file = io.open 'labels.txt'
  local list = {}
  while true do
    local line = file:read()
    if not line then break end
    table.insert(list, line)
  end
  return list
end


-- Converts an image from RGB to BGR format and subtracts mean
function preprocess(im, img_mean)
  local im3 = im*255
  channels = {'b','g','r'}
  -- RGB2BGR
  local im4 = im3:clone()
  im4[{1,{},{}}] = im3[{3,{},{}}]
  im4[{3,{},{}}] = im3[{1,{},{}}]

  for i,channel in ipairs(channels) do
    -- normalize each channel globall
    im4[{ i,{},{} }]:add(-img_mean[i])
  end

  return im4
end


local class_labels = load_classes()

if not paths.filep(opt.image) then
  os.execute('wget '..opt.image)
  opt.image = opt.image:match( "([^/]+)$" )
end


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

print('Loading ' .. paths.concat(opt.load, opt.network .. '_' .. opt.epoch .. '_Weights.t7') .. ' file')

local im = image.load(opt.image)

local img_mean=image.load(opt.mean):type('torch.DoubleTensor')

if (img_mean:size(1) ~= im:size(1)) or (img_mean:size(2) ~= im:size(2)) or (img_mean:size(3) ~= im:size(3)) then

   os.execute('python /home/ubuntu/p4/montessori/dev/tools/resize_image.py ' .. opt.image .. ' ' .. 'mod_' .. opt.image .. ' ' .. img_mean:size(2) .. ' ' .. img_mean:size(3) .. ' -c ' .. img_mean:size(1) .. ' -m ' .. opt.mode)   --path needs to be corrected for resize_image.py
   im = image.load('mod_' .. opt.image)
end



if opt.type == 'float' then
    print('==> switching to floats')
    torch.setdefaulttensortype('torch.FloatTensor')

elseif opt.type =='cuda' then
    require 'cunn'
    print('==> switching to CUDA')
    --model:cuda()
    torch.setdefaulttensortype('torch.CudaTensor')

end

local model = require('./Models/' .. opt.network)
local weights, gradients = model:getParameters()

local weights_filename = paths.concat(opt.load, opt.network .. '_' .. opt.epoch .. '_Weights.t7')
weights:copy(torch.load(weights_filename))

if opt.type =='cuda' then
  model:cuda()
end

-- as we want to classify, let's disable dropouts by enabling evaluation mode
model:evaluate()

-- Have to resize and convert from RGB to BGR and subtract mean
I = preprocess(im, img_mean)

local inputs

if opt.ccn2 == 'yes' then
  inputs = torch.Tensor(32, img_mean:size(1), img_mean:size(2), img_mean:size(3))

  for i=1,32 do
    inputs[i]=I
  end
else
  inputs = torch.Tensor(1, img_mean:size(1), img_mean:size(2), img_mean:size(3))
  inputs[1]=I
end

if opt.type == 'float' then
  inputs=inputs:float()
elseif opt.type =='cuda' then
  inputs=inputs:cuda()
end

local y = model:forward(inputs)

-- for the outputs of SoftMax layer sort them in decreasing order
val,classes = y[{1,{}}]:float():sort(true)

for i=1,5 do
  print('predicted class '..tostring(i)..': ', class_labels[classes[i]], math.exp(val[i])*100)
end



