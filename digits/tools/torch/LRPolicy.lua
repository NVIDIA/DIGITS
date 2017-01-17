-- Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.

local LRPolicy = torch.class('LRPolicy')

----------------------------------------------------------------------------------------
-- This file contains details of learning rate policies that are used in caffe.
-- Calculates and returns the current learning rate. The currently implemented learning rate
-- policies are as follows:
--    - fixed: always return base_lr.
--    - step: return base_lr * gamma ^ (floor(iter / step))
--    - exp: return base_lr * gamma ^ iter
--    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
--    - multistep: similar to step but it allows non uniform steps defined by
--      stepvalue
--    - poly: the effective learning rate follows a polynomial decay, to be
--      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
--    - sigmoid: the effective learning rate follows a sigmod decay
--      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
--------------------------------------------------------------------------------------------------



function LRPolicy:__init(...)
    local args = dok.unpack(
    {...},
    'LRPolicy','Initialize a learning rate policy',
    {arg='policy', type ='string', help='Learning rate policy',req=true},
    {arg='baselr', type ='number', help='Base learning rate',req=true},
    {arg='gamma', type = 'number', help='parameter to compute learning rate',req=false},
    {arg='power', type = 'number', help='parameter to compute learning rate', req = false},
    {arg='step_size', type = 'number', help='parameter to compute learning rate', req = false},
    {arg='max_iter', type = 'number', help='parameter to compute learning rate', req = false},
    {arg='step_values', type = 'table', help='parameter to compute learning rate. Useful only when the learning rate policy is multistep', req = false}
    )
    self.policy = args.policy
    self.baselr = args.baselr
    self.gamma = args.gamma
    self.power = args.power
    self.step_values = args.step_values
    self.max_iter = args.max_iter

    if self.policy == 'step' or self.policy == 'sigmoid' then
      self.step_size = self.step_values[1]     -- if the policy is not multistep, then even though multiple step values are provided as input, we will consider only the first value.
    elseif self.policy == 'multistep' then
      self.current_step = 1       -- this counter is important to take arbitrary steps
      self.stepvalue_size = #self.step_values
    end

end

function LRPolicy:GetLearningRate(iter)

  local rate=0
  local progress = 100 * (iter / self.max_iter)  -- expressed in percent units

  if self.policy == "fixed" then
    rate = self.baselr
  elseif self.policy == "step" then
    local current_step = math.floor(iter/self.step_size)
    rate = self.baselr * math.pow(self.gamma, current_step)
  elseif self.policy == "exp" then
    rate = self.baselr * math.pow(self.gamma, progress)
  elseif self.policy == "inv" then
    rate = self.baselr * math.pow(1 + self.gamma * progress, - self.power)
  elseif self.policy == "multistep" then
    if (self.current_step <= self.stepvalue_size and iter >= self.step_values[self.current_step]) then
      self.current_step = self.current_step + 1
    end
    rate = self.baselr * math.pow(self.gamma, self.current_step - 1);
  elseif self.policy == "poly" then
    rate = self.baselr * math.pow(1.0 - (iter / self.max_iter), self.power)
  elseif self.policy == "sigmoid" then
    rate = self.baselr * (1.0 / (1.0 + math.exp(self.gamma * (progress - 100*self.step_size/self.max_iter))));
  else
    --have to include additional comments
    print("Unknown learning rate policy: " .. self.policy)
  end

  return rate

end
