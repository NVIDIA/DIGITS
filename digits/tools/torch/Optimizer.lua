--[[
Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.

Copyright (c) 2004 Elad Hoffer

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
--]]

local Optimizer = torch.class('Optimizer')

function Optimizer:__init(...)
    xlua.require('torch',true)
    xlua.require('nn',true)
    local args = dok.unpack(
    {...},
    'Optimizer','Initialize an optimizer',
    {arg='Model', type ='table', help='Optimized model',req=true},
    {arg='Loss', type ='function', help='Loss function',req=true},
    {arg='Parameters', type = 'table', help='Model parameters - weights and gradients',req=false},
    {arg='OptFunction', type = 'function', help = 'Optimization function' ,req = true},
    {arg='OptState', type = 'table', help='Optimization configuration', default = {}, req=false},
    {arg='HookFunction', type = 'function', help='Hook function of type fun(y,yt,err)', req = false},
    {arg='lrPolicy', type = 'table', help='learning rate policy', req = true},
    {arg='LabelFunction', type = 'function', help = 'Label function', req = true}
    )
    self.Model = args.Model
    self.Loss = args.Loss
    self.Parameters = args.Parameters
    self.OptFunction = args.OptFunction
    self.OptState = args.OptState
    self.HookFunction = args.HookFunction
    self.lrPolicy = args.lrPolicy
    self.LabelFunction = args.LabelFunction

    if self.Parameters == nil then
        self.Parameters = {}
        self.Weights, self.Gradients = self.Model:getParameters()
    else
        self.Weights, self.Gradients = self.Parameters[1], self.Parameters[2]
    end
end

function Optimizer:optimize(x,yt)
    local f_eval = function()
        self.Model:zeroGradParameters()
        local y = self.Model:forward(x)
        -- get label
        label = self.LabelFunction(x,yt)
        local err = self.Loss:forward(y,label)
        local dE_dy = self.Loss:backward(y,label)
        local value = nil
        self.Model:backward(x, dE_dy)
        if self.HookFunction then
            value = self.HookFunction(y,label,err)
        end
        return err, self.Gradients
    end

    if self.lrPolicy.policy ~= 'torch_sgd' then
        self.OptState.learningRate = self.lrPolicy:GetLearningRate(self.OptState.evalCounter or 0)   --- here self.OptState.evalCounter = iter/stepsize
    end

    return value, self.OptState.learningRate, self.OptFunction(f_eval, self.Weights, self.OptState)

end

