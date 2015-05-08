-- Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.

require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'trepl'
require 'cutorch'
require 'lfs'

local dir_path = debug.getinfo(1,"S").source:match[[^@?(.*[\/])[^\/]-$]]
if dir_path ~= nil then
  package.path = dir_path .."?.lua;".. package.path
end

require 'Optimizer'
require 'LRPolicy'
require 'logmessage'
----------------------------------------------------------------------

opt = lapp[[
Usage details:
-a,--threads            (default 8)              number of threads
-b,--batchSize          (default 128)            batch size
-c,--learningRateDecay  (default 1e-6)           learning rate decay (in # samples)
-d,--devid              (default 1)              device ID (if using CUDA)
-e,--epoch              (number)                 number of epochs to train -1 for unbounded
-f,--shuffle            (default no)             shuffle records before train
-g,--mirror             (default no)             If this option is 'yes', then some of the images are randomly mirrored                                   
-i,--interval           (default 1)              number of train epochs to complete, to perform one validation
-k,--crop               (default no)             If this option is 'yes', all the images are randomly cropped into square image. And croplength is provided as --croplen parameter 
-l,--croplen            (default 0)              crop length. This is required parameter when crop option is provided
-m,--momentum           (default 0.9)            momentum
-n,--network	        (string)                 Model - must return valid network. Available - {lenet, googlenet, alexnet}
-o,--optimization       (default sgd)            optimization method
-p,--type               (default cuda)           float or cuda
-r,--learningRate       (default 0.001)          learning rate
-s,--save               (default results)        save directory
-t,--train              (string)                 location in which train db exists. 
-v,--validation         (default '')             location in which validation db exists. 
-w,--weightDecay        (default 1e-4)           L2 penalty on the weights 

--networkDirectory      (default '')             directory in which network exists
--mean                  (default mean.jpg)       mean file. Mean file is used to preprocess images and it is also required to get the details of image channel, height and width.
--subtractMean          (default yes)            If yes, subtracts the mean from images
--labels                (default labels.txt)     file contains label definitions
--snapshotPrefix        (default '')             prefix of the weights/snapshots
--snapshotInterval      (default 1)              specifies the training epochs to be completed before taking a snapshot
--useMeanPixel          (default 'no')           by default pixel-wise subtraction is done using the full mean matrix. If this option is 'yes' then mean pixel will be used instead of mean matrix

-q,--policy             (default torch_sgd)      Learning Rate Policy. Valid policies : fixed, step, exp, inv, multistep, poly, sigmoid and torch_sgd. Note: when power value is -1, then "inv" policy with "gamma" is similar to "torch_sgd" with "learningRateDecay".              
-h,--gamma              (default -1)             Required to calculate learning rate, when any of the following learning rate policies are used:  step, exp, inv, multistep & sigmoid                        
-j,--power              (default inf)            Required to calculate learning rate, when any of the following learning rate policies are used:  inv & poly  
-x,--stepvalues         (default '')             Required to calculate stepsize for the following learning rate policies:  step, multistep & sigmoid. Note: if it is 'step' or 'sigmoid' policy, then this parameter expects single value, if it is 'multistep' policy, then this parameter expects a string which has all the step values delimited by comma (ex: "10,25,45,80") 
]]


-- validate options
if opt.crop == 'yes' and opt.croplen == 0 then
    logmessage.display(2,'crop length is missing')
  return 
end

local stepvalues_list = {}

-- verify whether required learning rate parameters are provided to calculate learning rate when caffe-like learning rate policies are used
if opt.policy == 'fixed' or opt.policy == 'step' or opt.policy == 'exp' or opt.policy == 'inv' or opt.policy == 'multistep' or opt.policy == 'poly' or opt.policy == 'sigmoid' then

  if opt.policy == 'step' or opt.policy == 'exp' or opt.policy == 'inv' or opt.policy == 'multistep' or opt.policy == 'sigmoid' then
    if opt.gamma ==-1 then
      logmessage.display(2,'gamma parameter missing and is required to calculate learning rate when ' .. opt.policy .. ' learning rate policy is used')
      return
    end
  end

  if opt.policy == 'inv' or opt.policy == 'poly' then 
    if opt.power == math.huge then
      logmessage.display(2,'power parameter missing and is required to calculate learning rate when ' .. opt.policy .. ' learning rate policy is used')
      return
    end
  end

  if opt.policy == 'step' or opt.policy == 'multistep' or opt.policy == 'sigmoid' then
    if opt.stepvalues =='' then
      logmessage.display(2,'step parameter missing and is required to calculate learning rate when ' .. opt.policy .. ' learning rate policy is used')
      return
    else
          
      for i in string.gmatch(opt.stepvalues, '([^,]+)') do
        if tonumber(i) ~= nil then
          table.insert(stepvalues_list, tonumber(i))
        else
          logmessage.display(2,'invalid step parameter value : ' .. opt.stepvalues .. '. step parameter should contain only number. if there are more than one value, then the values should be delimited by comma. ex: "10" or "10,25,45,80"')
          return
          
        end
      end
    end
  end

elseif opt.policy ~= 'torch_sgd' then
  logmessage.display(2,'invalid learning rate policy - '.. opt.policy .. '. Valid policies : fixed, step, exp, inv, multistep, poly, sigmoid and torch_sgd')
  return 
end

if opt.useMeanPixel ~= 'yes' and opt.useMeanPixel ~= 'no' then
  logmessage.display(2,'invalid --useMeanPixel parameter value - '.. opt.useMeanPixel .. '. Only "yes" or "no" is allowed')
  return
end

if opt.useMeanPixel == 'yes' and opt.subtractMean ~= 'no' then
  opt.useMeanPixel = 'no'
  logmessage.display(0,'useMeanPixel parameter is not considered as subtractMean value is provided as "yes"')
end

torch.setnumthreads(opt.threads)
cutorch.setDevice(opt.devid)
----------------------------------------------------------------------
-- Model + Loss:

package.path =  paths.concat(opt.networkDirectory, "?.lua") ..";".. package.path
local model_filename = paths.concat(opt.networkDirectory, opt.network)
logmessage.display(0,'Loading Model: ' .. model_filename)
local model = require (opt.network)

local loss = nn.ClassNLLCriterion()

-- check whether ccn2 is used in network and then check whether given batchsize is valid or not
if ccn2 ~= nil then
  if opt.batchSize % 32 ~= 0 then
    logmessage.display(2,'invalid batch size : ' .. opt.batchSize .. '. Batch size should be multiple of 32 when ccn2 is used in the network')
    return
  end 
end

-- load  
local data = require 'data'

logmessage.display(0,'Loading mean tensor from '.. opt.mean ..' file')
local mean_t = data.loadMean(opt.mean, opt.useMeanPixel)

logmessage.display(0,'Loading label definitions from '.. opt.labels ..' file')
-- classes
local classes = data.loadLabels(opt.labels)

if classes == nil then
  logmessage.display(2,'labels file '.. opt.labels ..' not found')
  return
end

logmessage.display(0,'found ' .. #classes .. ' categories')


-- Set the seed of the random number generator to the current time in seconds.

if opt.mirror == 'yes' then
    torch.manualSeed(os.time())
    logmessage.display(0,'mirror option was selected, so during training for some of the random images, mirror view will be considered instead of original image view')
end
----------------------------------------------------------------------
------------- UTILITY FUNCTIONS ----------------------------

-- round function
function round(num, idp)
  local mult = 10^(idp or 0)
  return math.floor(num * mult + 0.5) / mult
end

----------------------------------------------------------------------
-- This matrix records the current confusion across classes
local confusion = optim.ConfusionMatrix(classes)

-- seperate validation matrix for validation data
local validation_confusion = nil
if opt.validation ~= '' then
    validation_confusion = optim.ConfusionMatrix(classes)
end


if opt.type == 'float' then
    logmessage.display(0,'switching to floats')
    torch.setdefaulttensortype('torch.FloatTensor')

elseif opt.type =='cuda' then
    require 'cunn'
    logmessage.display(0,'switching to CUDA')
    model:cuda()
    loss = loss:cuda()
    --torch.setdefaulttensortype('torch.CudaTensor')
end

-- create a directory, if not exists, to save all the snapshots
-- os.execute('mkdir -p ' .. paths.concat(opt.save))  -- commented this line, as os.execute command is not portable
if lfs.mkdir(paths.concat(opt.save)) then
    logmessage.display(0,'created a directory ' .. paths.concat(opt.save) .. ' to save all the snapshots')
end

-- open train lmdb file
logmessage.display(0,'opening train lmdb file: ' .. opt.train)
local train = DBSource:new(opt.train, opt.mirror, opt.crop, opt.croplen, mean_t, opt.subtractMean, true)
local trainSize = train:totalRecords()
logmessage.display(0,'found ' .. trainSize .. ' images in train db' .. opt.train)
local trainKeys
if opt.shuffle == 'yes' then
  logmessage.display(0,'loading all the keys from train db')
  trainKeys = train:getKeys()
end

local val, valSize, valKeys

if opt.validation ~= '' then
  logmessage.display(0,'opening validation lmdb file: ' .. opt.validation)
  -- for the images in validation dataset, no need to do random mirrorring.
  val = DBSource:new(opt.validation, 'no', opt.crop, opt.croplen, mean_t, opt.subtractMean, false)
  valSize = val:totalRecords()
  logmessage.display(0,'found ' .. valSize .. ' images in train db' .. opt.validation)
  if opt.shuffle == 'yes' then
    logmessage.display(0,'loading all the keys from validation db')
    valKeys = val:getKeys()
  end
end

-- validate "crop length" input parameter 
if opt.crop == 'yes' then
  if opt.croplen > train.ImageSizeY then
    logmessage.display(2,'invalid crop length! crop length ' .. opt.croplen .. ' is less than image width ' .. train.ImageSizeY)
    return
  elseif opt.croplen > train.ImageSizeX then
    logmessage.display(2,'invalid crop length! crop length ' .. opt.croplen .. ' is less than image height ' .. train.ImageSizeX)
    return
  end
end

--modifying total sizes of train and validation dbs to be the exact multiple of 32, when cc2 is used
if ccn2 ~= nil then
  if (trainSize % 32) ~= 0 then
    logmessage.display(1,'when ccn2 is used, total images should be the exact multiple of 32. In train db, as the total images  are ' .. trainSize .. ', skipped the last ' .. trainSize % 32 .. ' images from train db')
    trainSize = trainSize - (trainSize % 32)
  end
  if opt.validation ~= '' and (valSize % 32) ~=0 then
    logmessage.display(1,'when ccn2 is used, total images should be the exact multiple of 32. In validation db, as the total images are ' .. valSize .. ', skipped the last ' .. valSize % 32 .. ' images from validation db') 
    valSize = valSize - (valSize % 32)
  end
end

local lrpolicy = {}

if opt.policy ~= 'torch_sgd' then

    --resetting "learningRateDecay = 0", so that sgd.lua won't recalculates the learning rate 
    opt.learningRateDecay = 0

    local max_iterations = (math.ceil(trainSize/opt.batchSize))*opt.epoch
    --local stepsize = math.floor((max_iterations*opt.step/100)+0.5)    --adding 0.5 to round the value

    --converting stepsize percentages into values 
    for i=1,#stepvalues_list do
      stepvalues_list[i] = round(max_iterations*stepvalues_list[i]/100)
    end

    --initializing learning rate policy
    logmessage.display(0,'initializing the parameters for learning rate policy: ' .. opt.policy)
    lrpolicy = LRPolicy{
           policy = opt.policy,
           baselr = opt.learningRate,
           gamma = opt.gamma,
           power = opt.power,
           max_iter = max_iterations,
           step_values = stepvalues_list
    }

else 
    logmessage.display(0,'initializing the parameters for learning rate policy: ' .. opt.policy)
    lrpolicy = LRPolicy{
           policy = opt.policy,
           baselr = opt.learningRate
    }

end

local optimState = {
    learningRate = opt.learningRate,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.learningRateDecay
}

local function updateConfusion(y,yt)
    confusion:batchAdd(y,yt)
end

-- Optimization configuration
logmessage.display(0,'initializing the parameters for Optimizer')
local Weights,Gradients = model:getParameters()
local optimizer = Optimizer{
    Model = model,
    Loss = loss,
    OptFunction = optim.sgd,
    OptState = optimState,
    Parameters = {Weights, Gradients},
    HookFunction = updateConfusion,
    lrPolicy = lrpolicy
}

-- During training, loss rate should be displayed at max 8 times or for every 5000 images, whichever lower.
local logging_check = 0

if (math.ceil(trainSize/8)<5000) then
  logging_check = math.ceil(trainSize/8)
else
  logging_check = 5000
end  
logmessage.display(0,'During training. details will be logged after every ' .. logging_check .. ' images')


-- This variable keeps track of next epoch, when to perform validation.
local next_validation = opt.interval
logmessage.display(0,'Training epochs to be completed for each validation :  ' .. opt.interval)
local last_validation_epoch = 0

-- This variable keeps track of next epoch, when to save model weights.
local next_snapshot_save = opt.snapshotInterval
logmessage.display(0,'Training epochs to be completed before taking a snapshot : ' .. opt.snapshotInterval)
local last_snapshot_save_epoch = 0

local snapshot_prefix = ''

if opt.snapshotPrefix ~= '' then
    snapshot_prefix = opt.snapshotPrefix
else
    snapshot_prefix = opt.network
end

logmessage.display(0,'snapshots will be saved as ' .. snapshot_prefix .. '_<EPOCH>_Weights.t7')



-- Test function
local function Test()

    model:evaluate()
    local shuffle
    if opt.shuffle == 'yes' then
      shuffle = torch.randperm(valSize):cuda()
    end

    local NumBatches = 0
    local loss_sum = 0
    local inputs, targets

    if opt.shuffle == 'yes' then
      if opt.crop == 'yes' then
        inputs = torch.Tensor(opt.batchSize, val.ImageChannels, opt.croplen, opt.croplen)
      else
        inputs = torch.Tensor(opt.batchSize, val.ImageChannels, val.ImageSizeY, val.ImageSizeX)
      end
      targets = torch.Tensor(opt.batchSize)      
    end

    for t = 1,valSize,opt.batchSize do

      -- create mini batch
      NumBatches = NumBatches + 1

      if opt.shuffle == 'yes' then
        local ind = 0
        for i = t,math.min(t+opt.batchSize-1,valSize) do
          -- load new sample
          local input, target = val:getImgUsingKey(valKeys[shuffle[i]])
          ind = ind+1
          inputs[ind] = input
          targets[ind] = target
        end
        if ind < opt.batchSize then
          inputs = inputs:narrow(1,1,ind)
          targets = targets:narrow(1,1,ind)
        end

      else
          inputs,targets = val:nextBatch(math.min(valSize-t+1,opt.batchSize))
      end

      if opt.type =='cuda' then
          inputs=inputs:cuda()
          targets = targets:cuda()
      else 
          inputs=inputs:float()
      end

      local y = model:forward(inputs)
      local err = loss:forward(y,targets)
      loss_sum = loss_sum + err
      validation_confusion:batchAdd(y,targets)

      if math.fmod(NumBatches,50)==0 then
          collectgarbage()
      end
    end

    return (loss_sum/NumBatches)

    --xlua.progress(valSize, valSize)
end


-- Train function
local function Train(epoch)

    model:training()
    local shuffle=nil;
    if opt.shuffle == 'yes' then
      shuffle = torch.randperm(trainSize):cuda()
    end

    local NumBatches = 0
    local curr_images_cnt = 0
    local loss_sum = 0
    local loss_batches_cnt = 0
    local learningrate = 0
    local inputs, targets

    if opt.shuffle == 'yes' then
      if opt.crop == 'yes' then
        inputs = torch.Tensor(opt.batchSize, train.ImageChannels, opt.croplen, opt.croplen)
      else
        inputs = torch.Tensor(opt.batchSize, train.ImageChannels, train.ImageSizeY, train.ImageSizeX)
      end

      targets = torch.Tensor(opt.batchSize)      
    end

    for t = 1,trainSize,opt.batchSize do

      -- create mini batch
      NumBatches = NumBatches + 1
      if opt.shuffle == 'yes' then
        local ind = 0
        for i = t,math.min(t+opt.batchSize-1,trainSize) do
          -- load new sample
          local input, target = train:getImgUsingKey(trainKeys[shuffle[i]])
          ind = ind+1
          inputs[ind] = input   -- this is similar to inputs[i%batchSize]
          targets[ind] = target
        end
        -- if the final set of images are less than batch size, then resize inputs and targets tensors
        if ind < opt.batchSize then
          inputs = inputs:narrow(1,1,ind)
          targets = targets:narrow(1,1,ind)
        end
      else
          inputs,targets = train:nextBatch(math.min(trainSize-t+1,opt.batchSize))
      end

      if opt.type =='cuda' then
          inputs = inputs:cuda()
          targets = targets:cuda()
      else 
          inputs = inputs:float()
      end

      _,learningrate,_,trainerr = optimizer:optimize(inputs, targets)

      -- adding the loss values of each mini batch and also maintaining the counter for number of batches, so that average loss value can be found at the time of logging details
      loss_sum = loss_sum + trainerr[1]
      loss_batches_cnt = loss_batches_cnt + 1

      if math.fmod(NumBatches,50)==0 then
        collectgarbage()
      end

      local current_epoch = (epoch-1)+round((t+opt.batchSize)/trainSize,2)

      -- log details when required number of images are processed
      curr_images_cnt = curr_images_cnt + opt.batchSize
      if curr_images_cnt >= logging_check then
        logmessage.display(0, 'Training (epoch ' .. current_epoch .. '): loss = ' .. (loss_sum/loss_batches_cnt) .. ', lr = ' .. learningrate)
        curr_images_cnt = 0             -- For accurate values we may assign curr_images_cnt % logging_check to curr_images_cnt, instead of 0
        loss_sum = 0
        loss_batches_cnt = 0
      end

      if opt.validation ~= '' and current_epoch >= next_validation then
          validation_confusion:zero()
          val:reset()
          local avg_loss=Test()
          validation_confusion:updateValids()
          -- log details at the end of validation
          logmessage.display(0, 'Validation (epoch ' .. current_epoch .. '): loss = ' .. avg_loss .. ', accuracy = ' .. validation_confusion.totalValid)

          next_validation = (round(current_epoch/opt.interval) + 1) * opt.interval            -- To find next nearest epoch value that exactly divisible by opt.interval
          last_validation_epoch = current_epoch
          model:training()    -- to reset model to training
      end

      if current_epoch >= next_snapshot_save then
          local weights_filename = paths.concat(opt.save, snapshot_prefix .. '_' .. current_epoch .. '_Weights.t7')
          logmessage.display(0,'Snapshotting to ' .. weights_filename)
          torch.save(weights_filename, Weights)
          logmessage.display(0,'Snapshot saved - ' .. weights_filename)

          next_snapshot_save = (round(current_epoch/opt.snapshotInterval) + 1) * opt.snapshotInterval            -- To find next nearest epoch value that exactly divisible by opt.snapshotInterval
          last_snapshot_save_epoch = current_epoch
      end

    end

    -- display the progress at the end of epoch
    if curr_images_cnt > 0 then
      logmessage.display(0, 'Training (epoch ' .. epoch .. '): loss = ' .. (loss_sum/loss_batches_cnt) .. ', lr = ' .. learningrate)
    end

    --xlua.progress(trainSize, trainSize)

end


------------------------------

local epoch = 1

logmessage.display(0,'started training the model')

while epoch<=opt.epoch do
    local ErrTrain = 0
    train:reset()
    confusion:zero()
    Train(epoch)
    confusion:updateValids()
    ErrTrain = (1-confusion.totalValid)
    epoch = epoch+1
end


-- if required, perform validation at the end
if opt.validation ~= '' and opt.epoch > last_validation_epoch then
    validation_confusion:zero()
    val:reset()
    local avg_loss=Test()
    validation_confusion:updateValids()
    -- log details at the end of validation
    logmessage.display(0, 'Validation (epoch ' .. opt.epoch .. '): loss = ' .. avg_loss .. ', accuracy = ' .. validation_confusion.totalValid)
end

-- if required, save snapshot at the end
if opt.epoch > last_snapshot_save_epoch then
    local weights_filename = paths.concat(opt.save, snapshot_prefix .. '_' .. opt.epoch .. '_Weights.t7')
    logmessage.display(0,'Snapshotting to ' .. weights_filename)
    torch.save(weights_filename, Weights)
    logmessage.display(0,'Snapshot saved - ' .. weights_filename)
end

train:close()
if opt.validation ~= '' then  
  val:close()
end

--print(confusion)
