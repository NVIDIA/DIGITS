-- Copyright (c) 2015-2016, NVIDIA CORPORATION. All rights reserved.
require 'torch'
require 'pl'

opt = lapp[[
-n,--network (default '') path to network file
-o,--output (default '') output path of model_def.json
]]

local parameters = {
    ngpus =  0,
    nclasses = nil,
    inputShape = nil
}

local network = require (opt.network)(parameters)
local model   = network.model

local dir_path = debug.getinfo(1,"S").source:match[[^@?(.*[\/])[^\/]-$]]
if dir_path ~= nil then
    package.path = dir_path .."?.lua;".. package.path
end

local _ = require('moses')
local JSON = require "JSON"

function getChains(model)
  -- Add chain property to all layer of a model (to act as a unique ID)
  function addLink(layer,link,chain)
    chain = chain .. "_" .. link
    layer.chain = chain
    if layer.modules then
      for i=1,#layer.modules do
        local child = layer:get(i)
        link = link + 1
        addLink(child,link,chain)
      end
    end
  end

  addLink(model,0,"")
end


function getContainers(layers)
  -- Get array of index values for parent containers
  local containers = {}
  local layer_id = 0

  function getContainer(layer,container_id)
    layer_id = layer_id + 1
    table.insert(containers, container_id)
    if layer.modules then
      local id = layer_id
      for i=1, #layer.modules do
        child = layer:get(i)
        getContainer(child,id)
      end
    end
  end

  getContainer(layers[1],0)
  return containers
end

function layersToNodes(layers,containers)
  _.each(layers, function(i,layer)
    layer.index = i
    layer.container = layers[containers[i]]
  end)
  return layers
end

function getLeafNodes(leafNodes, obj)
  -- Get the leaf nodes of a given branch

  if not obj.isLast then
    _.each(obj.children, function(i,c) getLeafNodes(leafNodes,c) end)
  else
    table.insert(leafNodes,obj)
  end
end

function getType(node)
  if type(node.type) == "string" then
    return node.type
  else
    return torch.type(node)
  end
end

function getNodesAndLinks(nodes,links,obj)
  -- Get the links between each node
  local source  = obj
  if(obj.isLast ~= true) then
    _.each(obj.children, function(i,child)
      -- print(getType(child))
      local src = {index= source.index, type=getType(source)}
      local trg = {index= child.index, type=getType(child)}

      table.insert(links, {source=src, target=trg})
      if (_.indexOf(_.pluck(nodes,"index"),child.index)) then return end

      local node = {type=getType(child), chain=child.chain, index=child.index}
      table.insert(nodes,node)
      getNodesAndLinks(nodes,links,child)
    end)
  end
end

function toGraph(node)
  -- Convert Tree Structure to Graph Structure
  local links = {}
  local nodes = {}
  local graph = {nodes={}, links={} };

  local n = {type=getType(node), chain=node.chain, index=node.index}
  table.insert(graph.nodes,n)
  getNodesAndLinks(graph.nodes, links, node)

  graph.links = links

  return graph;
end

function isContainer(node)
  return node.modules
end

function isParallel(node)
  local type = torch.type(node)

  if string.match(type, "Concat") or string.match(type,"Parallel") or string.match(type,"DepthConcat") then
    return true
  else
    return false
  end
end
function isSequential(node)
  local type = torch.type(node)
  if string.match(type, "Sequential") then
    return true
  else
    return false
  end
end

function chainContents(node, nodes)
  -- Make the parent of each sibling its previous sibling

  node.children = {node.contents[1]};
  node.contents[1].parents = {node};
  local prevNode = node;

  _.each(node.contents, function(i,child)
    child.parents = {prevNode}
    prevNode = child

    if isContainer(child) then
      local exit = isParallel(child) and branchContents(child,nodes) or chainContents(child,nodes)
      prevNode = exit
      child = exit
      exit.isLast = false
    end

    local len = table.getn(node.contents)
    if (i ~= len) then child.children = {node.contents[i+1]} end
    if (i == len) then child.isLast = true end

  end)

  local exitIndex = prevNode.index + 0.5
  local exit = {index= exitIndex, type= "s-exit", children={}, parents={}, isLast=true}
  table.insert(nodes,exit)
  prevNode.children = {exit}
  prevNode.isLast = false
  exit.parents = {prevNode}

  return exit

end

function branchContents(node,nodes)
  -- Create branch structure, that terminates with the leaves
  -- of each branch joining together

  local leafNodes = {};
  node.children = {};

  -- Connect all of concats children to its parents
  local newParent = node.parents[1];
  newParent.children = {};

  _.each(node.contents, function(i,child)
    if isSequential(child) then chainContents(child,nodes) end
    child.parents = {newParent}
    table.insert(newParent.children, child)
  end)

  getLeafNodes(leafNodes, newParent)

  local exitIndex = _.max(_.pluck(leafNodes, "index")) + 0.4
  local exit = {index= exitIndex, type= "Concat", children={},parents={},isLast=true}
  _.each(leafNodes, function(i,leaf)
    leaf.children = {exit}
    leaf.isLast = false
    table.insert(exit.parents,leaf)
  end)
  return exit;
end

function removeNodes(nodes, type)
  _.each(_.filter(nodes, function(i,n) return getType(n) == type end), function(i,n)
    if _.isNil(n.parents) then return end

    local newParent = n.parents[1]
    local newChildren = n.children

    _.each(newChildren, function(i,c)
      table.insert(newParent.children,c)
      c.parents = {newParent}
    end)

    newParent.children = _.filter(newParent.children, function(i,c)
      return getType(c) ~= type
    end)

  end)
end

function getContainerContents(node,nodes)
  return _.filter(nodes, function(i,n)
    if _.isNil(n.container) then return false end
    return n.container.index == node.index
  end)
end

getChains(model)
local containers = getContainers(model:listModules())
local nodes = layersToNodes(model:listModules(),containers)

_.each(nodes, function(i,node)
  -- getContainerContents(node,nodes)
  node.contents = getContainerContents(node,nodes);
  -- if (node.type == "nn.Sequential") chainContents(node);
end)

chainContents(nodes[1], nodes);

-- Remove all exit blocks, and connect their children to their parents
removeNodes(nodes, "s-exit")

-- Remove all sequence entry blocks and attatch to children:
removeNodes(nodes, "nn.Sequential")

local graph = toGraph(nodes[1])

local file = io.open(opt.output, "w")
file:write(JSON:encode_pretty(graph))
file:close()
