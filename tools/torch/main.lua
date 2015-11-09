-- Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.

require 'torch'
require 'xlua'
require 'optim'
require 'pl'
require 'trepl'
require 'lfs'
require 'nn'

local dir_path = debug.getinfo(1,"S").source:match[[^@?(.*[\/])[^\/]-$]]
if dir_path ~= nil then
    package.path = dir_path .."?.lua;".. package.path
end

require 'Optimizer'
require 'LRPolicy'
require 'logmessage'

-- load utils
local utils = require 'utils'
----------------------------------------------------------------------

opt = lapp[[
Usage details:
-a,--threads            (default 8)              number of threads
-b,--batchSize (default 0) batch size
-c,--learningRateDecay (default 1e-6) learning rate decay (in # samples)
-d,--devid (default 1) device ID (if using CUDA)
-e,--epoch (default 1) number of epochs to train -1 for unbounded
-f,--shuffle (default no) shuffle records before train
-g,--mirror (default no) If this option is 'yes', then some of the images are randomly mirrored
-i,--interval (default 1) number of train epochs to complete, to perform one validation
-k,--crop (default no) If this option is 'yes', all the images are randomly cropped into square image. And croplength is provided as --croplen parameter
-l,--croplen (default 0) crop length. This is required parameter when crop option is provided
-m,--momentum (default 0.9) momentum
-n,--network (string) Model - must return valid network. Available - {lenet, googlenet, alexnet}
-o,--optimization (default sgd) optimization method
-p,--type (default cuda) float or cuda
-r,--learningRate (default 0.001) learning rate
-s,--save (default results) save directory
-t,--train (default '') location in which train db exists. This parameter may be omitted only if visualizeModel is 'yes'.
-v,--validation (default '') location in which validation db exists.
-w,--weightDecay (default 1e-4) L2 penalty on the weights

--train_labels (default '') location in which train labels db exists. Optional, use this if train db does not contain target labels.
--validation_labels (default '') location in which validation labels db exists. Optional, use this if validation db does not contain target labels.
--dbbackend (default 'lmdb') Specifies which DB backend was used to create datasets. Valid backends: hdf5, lmdb
--seed (default '') fixed input seed for repeatable experiments
--weights (default '') filename for weights of a model to use for fine-tuning
--retrain (default '') Specifies path to model to retrain with
--optimState (default '') Specifies path to an optimState to reload from
--randomState (default '') Specifies path to a random number state to reload from
--lrpolicyState (default '') Specifies path to a lrpolicy state to reload from
--networkDirectory (default '') directory in which network exists
--mean (default '') mean image file.
--subtractMean (default 'image') Select mean subtraction method. Possible values are 'image', 'pixel' or 'none'.
--labels (default '') file contains label definitions
--snapshotPrefix (default '') prefix of the weights/snapshots
--snapshotInterval (default 1) specifies the training epochs to be completed before taking a snapshot
--visualizeModel (default 'no') Visualize model. If this options is set to 'yes' no model will be trained.

-q,--policy (default torch_sgd) Learning Rate Policy. Valid policies : fixed, step, exp, inv, multistep, poly, sigmoid and torch_sgd. Note: when power value is -1, then "inv" policy with "gamma" is similar to "torch_sgd" with "learningRateDecay".
-h,--gamma (default -1) Required to calculate learning rate, when any of the following learning rate policies are used: step, exp, inv, multistep & sigmoid
-j,--power (default inf) Required to calculate learning rate, when any of the following learning rate policies are used: inv & poly
-x,--stepvalues (default '') Required to calculate stepsize for the following learning rate policies: step, multistep & sigmoid. Note: if it is 'step' or 'sigmoid' policy, then this parameter expects single value, if it is 'multistep' policy, then this parameter expects a string which has all the step values delimited by comma (ex: "10,25,45,80")
]]

-----------------------------------------------------------------------------------------------------------------------------
--Note: At present DIGITS supports only fine tuning, which means copying only the weights from pretrained model.
--
--To include "crash recovery" feature, we may need to save the below torch elements for every fixed duration (or) for every fixed epochs (for instance 30 minutes or 10 epochs).
--
-- trained model
-- SGD optim state
-- LRPolicy - this module helps in implementing caffe learning policies in Torch
-- Random number state
--
--And if the job was crashed, provide the saved backups using the command options (--retrain, --optimState, --randomState, --lrpolicyState) while restarting the job.
--
--Please refer to below links for more information about "crash recovery" feature:
-- 1) https://groups.google.com/forum/#!searchin/torch7/optimstate/torch7/uNxnrH-7C-4/pgIBdAFVaOYJ
-- 2) https://groups.google.com/forum/#!topic/torch7/fcy0-5v6M08
-- 3) https://groups.google.com/forum/#!searchin/torch7/optimstate/torch7/Gv1BiQoaIVA/HRnjRoegR38J
--
--Almost all the required routines are already implemented. Below are some remaining tasks,
-- 1) while recovering from crash, we should only consider the below options and discard all other inputs like epoch
-- --retrain, --optimState, --randomState, --lrpolicyState, --networkDirectory, --network, --save, --train, --validation, --mean, --labels, --snapshotPrefix
-- 2) We should also save and restore some information like epoch, batch size, snapshot interval, subtractMean, shuffle, mirror, crop, croplen
-- Precautions should be taken while restoring these options.
-----------------------------------------------------------------------------------------------------------------------------

----------------------------------------------------------------------
-- Initial parameter checks

-- Convert boolean options
opt.crop = opt.crop == 'yes' or false
opt.mirror = opt.mirror == 'yes' or false
opt.shuffle = opt.shuffle == 'yes' or false
opt.visualizeModel = opt.visualizeModel == 'yes' or false

-- Set the seed of the random number generator to the given number.
if opt.seed ~= '' then
    torch.manualSeed(tonumber(opt.seed))
end

-- validate options
if opt.crop and opt.croplen == 0 then
    logmessage.display(2,'crop length is missing')
    os.exit(-1)
end

local stepvalues_list = {}

-- verify whether required learning rate parameters are provided to calculate learning rate when caffe-like learning rate policies are used
if opt.policy == 'fixed' or opt.policy == 'step' or opt.policy == 'exp' or opt.policy == 'inv' or opt.policy == 'multistep' or opt.policy == 'poly' or opt.policy == 'sigmoid' then

    if opt.policy == 'step' or opt.policy == 'exp' or opt.policy == 'inv' or opt.policy == 'multistep' or opt.policy == 'sigmoid' then
        if opt.gamma ==-1 then
            logmessage.display(2,'gamma parameter missing and is required to calculate learning rate when ' .. opt.policy .. ' learning rate policy is used')
            os.exit(-1)
        end
    end

    if opt.policy == 'inv' or opt.policy == 'poly' then
        if opt.power == math.huge then
            logmessage.display(2,'power parameter missing and is required to calculate learning rate when ' .. opt.policy .. ' learning rate policy is used')
            os.exit(-1)
        end
    end

    if opt.policy == 'step' or opt.policy == 'multistep' or opt.policy == 'sigmoid' then
        if opt.stepvalues =='' then
            logmessage.display(2,'step parameter missing and is required to calculate learning rate when ' .. opt.policy .. ' learning rate policy is used')
            os.exit(-1)
        else

            for i in string.gmatch(opt.stepvalues, '([^,]+)') do
                if tonumber(i) ~= nil then
                    table.insert(stepvalues_list, tonumber(i))
                else
                    logmessage.display(2,'invalid step parameter value : ' .. opt.stepvalues .. '. step parameter should contain only number. if there are more than one value, then the values should be delimited by comma. ex: "10" or "10,25,45,80"')
                    os.exit(-1)

                end
            end
        end
    end

elseif opt.policy ~= 'torch_sgd' then
    logmessage.display(2,'invalid learning rate policy - '.. opt.policy .. '. Valid policies : fixed, step, exp, inv, multistep, poly, sigmoid and torch_sgd')
    os.exit(-1)
end

if opt.retrain ~= '' and opt.weights ~= '' then
    logmessage.display(2,"Both '--retrain' and '--weights' options cannot be used at the same time.")
    os.exit(-1)
end

if opt.randomState ~= '' and opt.seed ~= '' then
    logmessage.display(2,"Both '--randomState' and '--seed' options cannot be used at the same time.")
    os.exit(-1)
end

torch.setnumthreads(opt.threads)

----------------------------------------------------------------------
-- Open Data sources:
-- mean tensor,
-- training database and optionally: labels,
-- optionally: validation database and labels.

local data = require 'data'

local meanTensor
if opt.subtractMean ~= 'none' then
    assert(opt.mean ~= '', 'subtractMean parameter not set to "none" yet mean image path is unset')
    logmessage.display(0,'Loading mean tensor from '.. opt.mean ..' file')
    meanTensor = data.loadMean(opt.mean, opt.subtractMean == 'pixel')
end

local classes
local confusion
local validation_confusion

if opt.labels ~= '' then
    logmessage.display(0,'Loading label definitions from '.. opt.labels ..' file')
    -- classes
    classes = data.loadLabels(opt.labels)

    if classes == nil then
        logmessage.display(2,'labels file '.. opt.labels ..' not found')
        os.exit(-1)
    end

    -- This matrix records the current confusion across classes
    confusion = optim.ConfusionMatrix(classes)

    -- separate validation matrix for validation data
    validation_confusion = nil
    if opt.validation ~= '' then
        validation_confusion = optim.ConfusionMatrix(classes)
    end

    logmessage.display(0,'found ' .. #classes .. ' categories')
end

logmessage.display(0,'creating data readers')

-- DataLoader objects take care of loading data from
-- databases. Data loading and tensor manipulations
-- (e.g. cropping, mean subtraction, mirroring) are
-- performed from separate threads
local trainDataLoader, trainSize, inputTensorShape
local validationDataLoader, valSize

if opt.train ~= '' then
    -- create data loader for training dataset
    trainDataLoader = DataLoader:new(4, -- num threads
                                     package.path,
                                     opt.dbbackend, opt.train, opt.train_labels,
                                     opt.mirror, meanTensor,
                                     true, -- train
                                     opt.shuffle,
                                     classes ~= nil -- whether this is a classification task
                                     )
    -- retrieve info from train DB (number of records and shape of input tensors)
    trainSize, inputTensorShape = trainDataLoader:getInfo()
    logmessage.display(0,'found ' .. trainSize .. ' images in train db' .. opt.train)
    if opt.validation ~= '' then
        local shape
        validationDataLoader = DataLoader:new(4, -- num threads
                                              package.path,
                                              opt.dbbackend, opt.validation, opt.validation_labels,
                                              false, -- no need to do random mirrorring
                                              meanTensor,
                                              false, -- train
                                              false, -- shuffle
                                              classes ~= nil -- whether this is a classification task
                                              )
        valSize, shape = validationDataLoader:getInfo()
        logmessage.display(0,'found ' .. valSize .. ' images in train db' .. opt.validation)
    end
else
    assert(opt.visualizeModel, 'Train DB should be specified')
end

-- update inputTensorShape if crop length specified
-- this is necessary as the inputTensorShape is provided
-- below to the user-defined function that defines the network
if opt.crop and inputTensorShape then
    inputTensorShape[2] = opt.croplen
    inputTensorShape[3] = opt.croplen
end

----------------------------------------------------------------------
-- Model + Loss:
-- this is where we retrieve the network definition
-- from the user-defined function

package.path = paths.concat(opt.networkDirectory, "?.lua") ..";".. package.path
logmessage.display(0,'Loading network definition from ' .. paths.concat(opt.networkDirectory, opt.network))
local network_func = require (opt.network)
assert(type(network_func)=='function', "Network definition should return a Lua function - see documentation")
local parameters = {
        ngpus = (opt.type == 'cuda') and 1 or 0,
        inputShape = inputTensorShape,
    }
network = network_func(parameters)
local model = network.model

-- if the loss criterion was not defined in the network
-- use nn.ClassNLLCriterion() by default
local loss = network.loss or nn.ClassNLLCriterion()

-- if the crop length was not defined on command line then
-- check if the network defined a preferred crop length
if not opt.crop and network.croplen then
    opt.crop = true
    opt.croplen = network.croplen
end

-- if batch size was not specified on command line then check
-- whether the network defined a preferred batch size (there
-- can be separate batch sizes for the training and validation
-- sets)
local trainBatchSize
local validationBatchSize
if opt.batchSize==0 then
    local defaultBatchSize = 16
    trainBatchSize = network.trainBatchSize or defaultBatchSize
    validationBatchSize = network.validationBatchSize or defaultBatchSize
else
    trainBatchSize = opt.batchSize
    validationBatchSize = opt.batchSize
end
logmessage.display(0,'Train batch size is '.. trainBatchSize .. ' and validation batch size is ' .. validationBatchSize)

-- if we were instructed to print a visualization of the model,
-- do it now and return immediately
if opt.visualizeModel then
    logmessage.display(0,'Network definition:')
    print('\nModel: \n' .. model:__tostring())
    print('\nCriterion: \n' .. loss:__tostring())
    logmessage.display(0,'Network definition ends')
    os.exit(-1)
end

if opt.mirror then
    logmessage.display(0,'mirror option was selected, so during training for some of the random images, mirror view will be considered instead of original image view')
end

-- NOTE: currently randomState option wasn't used in DIGITS. This option was provided to be used from command line, if required.
-- load random number state from backup
if opt.randomState ~= '' then
    if paths.filep(opt.randomState) then
        logmessage.display(0,'Loading random number state - ' .. opt.randomState)
        torch.setRNGState(torch.load(opt.randomState))
    else
        logmessage.display(2,'random number state not found: ' .. opt.randomState)
        os.exit(-1)
    end
end

----------------------------------------------------------------------

-- NOTE: currently retrain option wasn't used in DIGITS. This option was provided to be used from command line, if required.
-- If preloading option is set, preload existing models appropriately
if opt.retrain ~= '' then
    if paths.filep(opt.retrain) then
        logmessage.display(0,'Loading pretrained model - ' .. opt.retrain)
        model = torch.load(opt.retrain)
    else
        logmessage.display(2,'Pretrained model not found: ' .. opt.retrain)
        os.exit(-1)
    end
end

if classes then
    -- for classification networks, and for the user's convenience, adjust
    -- last layer to match number of classes
    utils.correctFinalOutputDim(model, #classes)
end

logmessage.display(0,'Network definition: \n' .. model:__tostring__())
logmessage.display(0,'Network definition ends')

if opt.type == 'float' then
    logmessage.display(0,'switching to floats')
    torch.setdefaulttensortype('torch.FloatTensor')
    model:float()
    loss = loss:float()

elseif opt.type =='cuda' then
    require 'cunn'
    require 'cutorch'
    cutorch.setDevice(opt.devid)
    logmessage.display(0,'switching to CUDA')
    model:cuda()
    loss = loss:cuda()
    --torch.setdefaulttensortype('torch.CudaTensor')
end

local Weights,Gradients = model:getParameters()
-- If weights option is set, preload weights from existing models appropriately
if opt.weights ~= '' then
    if paths.filep(opt.weights) then
        logmessage.display(0,'Loading weights from pretrained model - ' .. opt.weights)
        Weights:copy(torch.load(opt.weights))
    else
        logmessage.display(2,'Weight file for pretrained model not found: ' .. opt.weights)
        os.exit(-1)
    end
end

-- create a directory, if not exists, to save all the snapshots
-- os.execute('mkdir -p ' .. paths.concat(opt.save)) -- commented this line, as os.execute command is not portable
if lfs.mkdir(paths.concat(opt.save)) then
    logmessage.display(0,'created a directory ' .. paths.concat(opt.save) .. ' to save all the snapshots')
end

-- validate "crop length" input parameter
if opt.crop and inputTensorShape then
    -- make sure crop length is not bigger than image height or width
    opt.croplen = math.min(opt.croplen, inputTensorShape[2], inputTensorShape[3])
    -- set crop length in data readers
    trainDataLoader:setCropLen(opt.croplen)
    if validationDataLoader then
        validationDataLoader:setCropLen(opt.croplen)
    end
end

--modifying total sizes of train and validation dbs to be the exact multiple of 32, when cc2 is used
if ccn2 ~= nil then
    if (trainSize % 32) ~= 0 then
        logmessage.display(1,'when ccn2 is used, total images should be the exact multiple of 32. In train db, as the total images are ' .. trainSize .. ', skipped the last ' .. trainSize % 32 .. ' images from train db')
        trainSize = trainSize - (trainSize % 32)
    end
    if opt.validation ~= '' and (valSize % 32) ~=0 then
        logmessage.display(1,'when ccn2 is used, total images should be the exact multiple of 32. In validation db, as the total images are ' .. valSize .. ', skipped the last ' .. valSize % 32 .. ' images from validation db')
        valSize = valSize - (valSize % 32)
    end
end

--initializing learning rate policy
logmessage.display(0,'initializing the parameters for learning rate policy: ' .. opt.policy)

local lrpolicy = {}
if opt.policy ~= 'torch_sgd' then

    local max_iterations = (math.ceil(trainSize/trainBatchSize))*opt.epoch
    --local stepsize = math.floor((max_iterations*opt.step/100)+0.5) --adding 0.5 to round the value

    if max_iterations < #stepvalues_list then
        logmessage.display(1,'maximum iterations (i.e., ' .. max_iterations .. ') is less than provided step values count (i.e, ' .. #stepvalues_list .. '), so learning rate policy is reset to "step" policy with the step value 1.')
        opt.policy = 'step'
        stepvalues_list[1] = 1
    else
        -- converting stepsize percentages into values
        for i=1,#stepvalues_list do
            stepvalues_list[i] = utils.round(max_iterations*stepvalues_list[i]/100)

            -- avoids 'nan' values during learning rate calculation
            if stepvalues_list[i] == 0 then
                stepvalues_list[i] = 1
            end
        end
    end

    lrpolicy = LRPolicy{
        policy = opt.policy,
        baselr = opt.learningRate,
        gamma = opt.gamma,
        power = opt.power,
        max_iter = max_iterations,
        step_values = stepvalues_list
    }

else
    lrpolicy = LRPolicy{
        policy = opt.policy,
        baselr = opt.learningRate
    }

end

-- NOTE: currently lrpolicyState option wasn't used in DIGITS. This option was provided to be used from command line, if required.
if opt.lrpolicyState ~= '' then
    if paths.filep(opt.lrpolicyState) then
        logmessage.display(0,'Loading lrpolicy state from file: ' .. opt.lrpolicyState)
        lrpolicy = torch.load(opt.lrpolicyState)
    else
        logmessage.display(2,'lrpolicy state file not found: ' .. opt.lrpolicyState)
        os.exit(-1)
    end
end

--resetting "learningRateDecay = 0", so that sgd.lua won't recalculates the learning rate
if lrpolicy.policy ~= 'torch_sgd' then
    opt.learningRateDecay = 0
end

local optimState = {
    learningRate = opt.learningRate,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.learningRateDecay
}

-- NOTE: currently optimState option wasn't used in DIGITS. This option was provided to be used from command line, if required.
if opt.optimState ~= '' then
    if paths.filep(opt.optimState) then
        logmessage.display(0,'Loading optimState from file: ' .. opt.optimState)
        optimState = torch.load(opt.optimState)

        -- this makes sure that sgd.lua won't recalculates the learning rate while using learning rate policy
        if lrpolicy.policy ~= 'torch_sgd' then
            optimState.learningRateDecay = 0
        end
    else
        logmessage.display(1,'Optim state file not found: ' .. opt.optimState) -- if optim state file isn't found, notify user and continue training
    end
end

local function updateConfusion(y,yt)
    if confusion ~= nil then
        confusion:batchAdd(y,yt)
    end
end

-- Optimization configuration
logmessage.display(0,'initializing the parameters for Optimizer')
local optimizer = Optimizer{
    Model = model,
    Loss = loss,
    --OptFunction = optim.sgd,
    OptFunction = _G.optim[opt.optimization],
    OptState = optimState,
    Parameters = {Weights, Gradients},
    HookFunction = updateConfusion,
    lrPolicy = lrpolicy,
    LabelFunction = network.labelHook or function (input,dblabel) return dblabel end,
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
logmessage.display(0,'Training epochs to be completed for each validation : ' .. opt.interval)
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

-- epoch value will be calculated for every batch size. To maintain unique epoch value between batches, it needs to be rounded to the required number of significant digits.
local epoch_round = 0 -- holds the required number of significant digits for round function.
local tmp_batchsize = trainBatchSize
while tmp_batchsize <= trainSize do
    tmp_batchsize = tmp_batchsize * 10
    epoch_round = epoch_round + 1
end
logmessage.display(0,'While logging, epoch value will be rounded to ' .. epoch_round .. ' significant digits')

logmessage.display(0,'Model weights will be saved as ' .. snapshot_prefix .. '_<EPOCH>_Weights.t7')

--[[ -- NOTE: uncomment this block when "crash recovery" feature was implemented
logmessage.display(0,'model, lrpolicy, optim state and random number states will be saved for recovery from crash')
logmessage.display(0,'model will be saved as ' .. snapshot_prefix .. '_<EPOCH>_model.t7')
logmessage.display(0,'optim state will be saved as optimState_<EPOCH>.t7')
logmessage.display(0,'random number state will be saved as randomState_<EPOCH>.t7')
logmessage.display(0,'LRPolicy state will be saved as lrpolicy_<EPOCH>.t7')
--]]

-- NOTE: currently this routine wasn't used in DIGITS.
-- This routine takes backup of model, optim state, LRPolicy and random number state
local function backupforrecovery(backup_epoch)
    -- save model
    local filename = paths.concat(opt.save, snapshot_prefix .. '_' .. backup_epoch .. '_model.t7')
    logmessage.display(0,'Saving model to ' .. filename)
    utils.cleanupModel(model)
    torch.save(filename, model)
    logmessage.display(0,'Model saved - ' .. filename)

    --save optim state
    filename = paths.concat(opt.save, 'optimState_' .. backup_epoch .. '.t7')
    logmessage.display(0,'optim state saving to ' .. filename)
    torch.save(filename, optimState)
    logmessage.display(0,'optim state saved - ' .. filename)

    --save random number state
    filename = paths.concat(opt.save, 'randomState_' .. backup_epoch .. '.t7')
    logmessage.display(0,'random number state saving to ' .. filename)
    torch.save(filename, torch.getRNGState())
    logmessage.display(0,'random number state saved - ' .. filename)

    --save lrPolicy state
    filename = paths.concat(opt.save, 'lrpolicy_' .. backup_epoch .. '.t7')
    logmessage.display(0,'lrpolicy state saving to ' .. filename)
    torch.save(filename, optimizer.lrPolicy)
    logmessage.display(0,'lrpolicy state saved - ' .. filename)
end


-- Validation function
local function Validation(dataLoader, nn_model, epoch)

    -- switch model to evaluation mode
    nn_model:evaluate()

    local NumBatches = 0
    local loss_sum = 0
    local inputs, targets
    local dataLoaderIdx = 1
    local data = {}

    local t = 1
    while t <= valSize do

        -- create mini batch
        NumBatches = NumBatches + 1

        while dataLoader:acceptsjob() do
            local dataBatchSize = math.min(valSize-dataLoaderIdx+1,validationBatchSize)
            if dataBatchSize > 0 then
                dataLoader:scheduleNextBatch(dataBatchSize, dataLoaderIdx, data, true)
                dataLoaderIdx = dataLoaderIdx + dataBatchSize
            else break end
        end

        -- wait for next data loader job to complete
        dataLoader:waitNext()

        -- get data from last load job
        local thisBatchSize = data.batchSize
        inputs = data.inputs
        targets = data.outputs

        if inputs then
            if opt.type =='cuda' then
                inputs=inputs:cuda()
                targets = targets:cuda()
            else
                inputs=inputs:float()
                targets = targets:float()
            end

            local y = model:forward(inputs)
            local labels = network.labelHook and network.labelHook(inputs, targets) or targets
            local err = loss:forward(y,labels)
            loss_sum = loss_sum + err
            if validation_confusion then
                validation_confusion:batchAdd(y,labels)
            end

            if math.fmod(NumBatches,50)==0 then
                collectgarbage()
            end

            t = t + thisBatchSize
        else
            -- failed to read from database (possibly due to disabled thread)
            dataLoaderIdx = dataLoaderIdx - data.batchSize
        end
    end

    return (loss_sum/NumBatches)

    --xlua.progress(valSize, valSize)
end

-- Train function
local function Train(epoch, dataLoader)

    model:training()

    local NumBatches = 0
    local curr_images_cnt = 0
    local loss_sum = 0
    local loss_batches_cnt = 0
    local learningrate = 0
    local inputs, targets

    local dataLoaderIdx = 1

    local data = {}

    local t = 1
    while t <= trainSize do

        while dataLoader:acceptsjob() do
            local dataBatchSize = math.min(trainSize-dataLoaderIdx+1,trainBatchSize)
            if dataBatchSize > 0 then
                dataLoader:scheduleNextBatch(dataBatchSize, dataLoaderIdx, data, true)
                dataLoaderIdx = dataLoaderIdx + dataBatchSize
            else break end
        end

        NumBatches = NumBatches + 1

        -- wait for next data loader job to complete
        dataLoader:waitNext()

        -- get data from last load job
        local thisBatchSize = data.batchSize
        inputs = data.inputs
        targets = data.outputs

        if inputs then
            --[=[
            -- print some statistics, show input in iTorch

            if t%1024==1 then
                print(string.format("input mean=%f std=%f",inputs:mean(),inputs:std()))
                for idx=1,thisBatchSize do
                    print(classes[targets[idx]])
                end
                if itorch then
                    itorch.image(inputs)
                end
            end
            --]=]

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

            local current_epoch = (epoch-1)+utils.round((math.min(t+trainBatchSize-1,trainSize))/trainSize, epoch_round)

            -- log details on first iteration, or when required number of images are processed
            curr_images_cnt = curr_images_cnt + thisBatchSize
            if (epoch==1 and t==1) or curr_images_cnt >= logging_check then
                logmessage.display(0, 'Training (epoch ' .. current_epoch .. '): loss = ' .. (loss_sum/loss_batches_cnt) .. ', lr = ' .. learningrate)
                curr_images_cnt = 0 -- For accurate values we may assign curr_images_cnt % logging_check to curr_images_cnt, instead of 0
                loss_sum = 0
                loss_batches_cnt = 0
            end

            if opt.validation ~= '' and current_epoch >= next_validation then
                if validation_confusion ~= nil then
                    validation_confusion:zero()
                end
                local avg_loss=Validation(validationDataLoader, model, current_epoch)
                 -- log details at the end of validation
                if validation_confusion ~= nil then
                    validation_confusion:updateValids()
                    logmessage.display(0, 'Validation (epoch ' .. current_epoch .. '): loss = ' .. avg_loss .. ', accuracy = ' .. validation_confusion.totalValid)
                else
                    logmessage.display(0, 'Validation (epoch ' .. current_epoch .. '): loss = ' .. avg_loss )
                end

                next_validation = (utils.round(current_epoch/opt.interval) + 1) * opt.interval -- To find next nearest epoch value that exactly divisible by opt.interval
                last_validation_epoch = current_epoch
                model:training() -- to reset model to training
            end

            if current_epoch >= next_snapshot_save then
                -- save weights
                local filename = paths.concat(opt.save, snapshot_prefix .. '_' .. current_epoch .. '_Weights.t7')
                logmessage.display(0,'Snapshotting to ' .. filename)
                torch.save(filename, Weights)
                logmessage.display(0,'Snapshot saved - ' .. filename)

                next_snapshot_save = (utils.round(current_epoch/opt.snapshotInterval) + 1) * opt.snapshotInterval -- To find next nearest epoch value that exactly divisible by opt.snapshotInterval
                last_snapshot_save_epoch = current_epoch
            end

            t = t + thisBatchSize
        else
            -- failed to read from database (possibly due to disabled thread)
            dataLoaderIdx = dataLoaderIdx - data.batchSize
        end

    end

    --xlua.progress(trainSize, trainSize)

end

------------------------------

local epoch = 1

logmessage.display(0,'started training the model')

-- run an initial validation before the first train epoch
if opt.validation ~= '' then
    model:evaluate()
    if validation_confusion ~= nil then
        validation_confusion:zero()
    end
    local avg_loss=Validation(validationDataLoader, model, 0)
    -- log details at the end of validation
    if validation_confusion ~= nil then
        validation_confusion:updateValids()
        logmessage.display(0, 'Validation (epoch ' .. epoch-1 .. '): loss = ' .. avg_loss .. ', accuracy = ' .. validation_confusion.totalValid)
    else
        logmessage.display(0, 'Validation (epoch ' .. epoch-1 .. '): loss = ' .. avg_loss )
    end
    model:training() -- to reset model to training
end

while epoch<=opt.epoch do
    local ErrTrain = 0
    if confusion ~= nil then
        confusion:zero()
    end
    Train(epoch, trainDataLoader)
    if confusion ~= nil then
        confusion:updateValids()
        --print(confusion)
        ErrTrain = (1-confusion.totalValid)
    end
    epoch = epoch+1
end

-- if required, perform validation at the end
if opt.validation ~= '' and opt.epoch > last_validation_epoch then
    if validation_confusion ~= nil then
        validation_confusion:zero()
    end
    local avg_loss=Validation(validationDataLoader, model, opt.epoch)
    -- log details at the end of validation
    if validation_confusion ~= nil then
        validation_confusion:updateValids()
        logmessage.display(0, 'Validation (epoch ' .. opt.epoch .. '): loss = ' .. avg_loss .. ', accuracy = ' .. validation_confusion.totalValid)
    else
        logmessage.display(0, 'Validation (epoch ' .. opt.epoch .. '): loss = ' .. avg_loss )
    end
end

-- if required, save snapshot at the end
if opt.epoch > last_snapshot_save_epoch then
    local filename = paths.concat(opt.save, snapshot_prefix .. '_' .. opt.epoch .. '_Weights.t7')
    logmessage.display(0,'Snapshotting to ' .. filename)
    torch.save(filename, Weights)
    logmessage.display(0,'Snapshot saved - ' .. filename)
end

-- close databases
trainDataLoader:close()
if opt.validation ~= '' then
    validationDataLoader:close()
end

-- enforce clean exit
os.exit(0)
