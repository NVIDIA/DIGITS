-- Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
opt = lapp[[
-n,--network (default '') path to network file
-o,--output (default '') output path of model_def.json
]]

print(opt.network)

local parameters = {
    ngpus =  0,
    nclasses = nil,
    inputShape = nil
}

local network = require (opt.network)(parameters)
local model   = network.model

nodeIndex = 0

function makeLine(depth)
  str = " "
  for i=1,2*depth do
    str = str .. " "
  end
  return str
end

function asJSON(nodeIndex,chain,node, parent)
  return '\n {"index": ' .. nodeIndex .. ', "chain": "' .. chain .. '", "type": "' .. torch.type(node) .. '", "container": { "index": ' .. parent['index'] .. ', "type": "' .. parent['type'] .. '" } },'
end

function cleanType(node)
  if torch.type(node) == "nn.Sequential" then
    return "s"
  elseif torch.type(node) == "nn.Concat" then
    return "c"
  end
end

function getChildren(node, link, chain, parent)
  chain = chain .. "_" .. link

  line = line .. asJSON(nodeIndex,chain,node, parent)

  local parent = {index=nodeIndex, type= cleanType(node)}

  nodeIndex = nodeIndex + 1

  if parent["type"] == "s" or parent["type"] == "c" then
    for i=1,#node.modules do
      child = node:get(i)
      link = link + 1
      getChildren(child,link,chain, parent)
    end
  end
end

line = "["
getChildren(model,0, "", {index=-1, type="nil"})
line = string.sub(line,0,-2) .. " \n ]"


local file = io.open(opt.output, "w")
file:write(line)
file:close()
