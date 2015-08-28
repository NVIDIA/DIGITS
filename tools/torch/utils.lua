-- Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.

require 'torch'   -- torch
require 'image'   -- for color transforms

package.path = debug.getinfo(1, "S").source:match[[^@?(.*[\/])[^\/]-$]] .."?.lua;".. package.path

require 'logmessage'

------------- UTILITY FUNCTIONS ----------------------------

local utilsClass={}

-- round function
local function round(num, idp)
  local mult = 10^(idp or 0)
  return math.floor(num * mult + 0.5) / mult
end

-- Currently zeroDataSize() and cleanupModel() routines aren't used but in future while implementing "recovery from crash" feature we may need to use these routines to clean the model before saving. This decreases the size of model by 80%.
function zeroDataSize(data)
  if type(data) == 'table' then
    for i = 1, #data do
      data[i] = zeroDataSize(data[i])
    end
  elseif type(data) == 'userdata' then
    data = torch.Tensor():typeAs(data)
  end
  return data
end

-- Resize the output, gradInput, etc temporary tensors to zero (so that the on disk size is smaller)
function cleanupModel(node)
  if node.output ~= nil then
    node.output = zeroDataSize(node.output)
  end
  if node.gradInput ~= nil then
    node.gradInput = zeroDataSize(node.gradInput)
  end
  if node.finput ~= nil then
    node.finput = zeroDataSize(node.finput)
  end
  -- Recurse on nodes with 'modules'
  if (node.modules ~= nil) then
    if (type(node.modules) == 'table') then
      for i = 1, #node.modules do
        local child = node.modules[i]
        cleanupModel(child)
      end
    end
  end

  -- Clear the references to the spatial convolution outputs as well
  if _spatial_convolution_mm_out ~= nil then
    _spatial_convolution_mm_out = {}
  end

  if _spatial_convolution_mm_gradout ~= nil then
    _spatial_convolution_mm_gradout = {}
  end

  collectgarbage()
end

utilsClass.round = round
utilsClass.cleanupModel = cleanupModel

--[[
Resizes an image and returns it as a np.array
Arguments:
image -- a PIL.Image or numpy.ndarray
height -- height of new image
width -- width of new image
Keyword Arguments:
channels -- channels of new image (stays unchanged if not specified)
resize_mode -- can be crop, squash, fill or half_crop
--]]
function utilsClass.resizeImage(img, height, width, channels,resize_mode)
    if resize_mode == nil then
        resize_mode = 'squash'
    elseif resize_mode ~= 'crop' and resize_mode ~= 'squash' and resize_mode ~= 'fill' and resize_mode ~= 'half_crop' then
        logmessage.display(0,'resize_mode ' .. resize_mode .. ' not supported')
    end

    if channels ~=nil and channels ~=3 and channels ~=1 then
        logmessage.display(0,'unsupported number of channels: ' .. channels)
    end

    --#TODO handle transparent images
    if img:size(1) == 2 then

    elseif img:size(1) == 4 then

    end

    if channels ~= nil and img:size(1) ~= channels then
        if img:size(1) == 3 and channels == 1 then
            img = image.rgb2y(img)
        elseif img:size(1) == 1 and channels == 3 then
            local dst = torch.Tensor(3,img:size(2),img:size(3)):type(img:type())
	    for i=1,3 do
                dst[{ i,{},{} }]:copy(img)
            end
	    img = dst
        end
    end
    -- No need to resize
    if img:size(2) == height and img:size(3) == width then
        return img
    end
    -- Resize
    width_ratio = img:size(3) / width
    height_ratio = img:size(2) / height
    if resize_mode == 'squash' or width_ratio == height_ratio then
        return image.scale(img,width,height)    
    elseif resize_mode == 'crop' then
        -- resize to smallest of ratios (relatively larger image), keeping aspect ratio
        if width_ratio > height_ratio then
            resize_height = height
            resize_width = round(img:size(3) / height_ratio)
        else
            resize_width = width
            resize_height = round(img:size(2) / width_ratio)
 	end
        img = image.scale(img,resize_width,resize_height)

	-- chop off ends of dimension that is still too long
	if width_ratio > height_ratio then
	    start = round((resize_width-width)/2.0)  
	    return image.crop(img, start, 0, start+width, img:size(2))
        else
	    start = round((resize_height-height)/2.0)
	    return image.crop(img, 0, start, img:size(3), start+height)
        end
    else 
	if resize_mode == 'fill' then
            -- resize to biggest of ratios (relatively smaller image), keeping aspect ratio
            if width_ratio > height_ratio then
                resize_width = width
                resize_height = round(img:size(2) / width_ratio)
                if (height - resize_height) % 2 == 1 then
                    resize_height = resize_height + 1
                end
            else
                resize_height = height
                resize_width = round(img:size(3) / height_ratio)
                if (width - resize_width) % 2 == 1 then
                    resize_width = resize_width + 1
                end
            end
            img = image.scale(img,resize_width,resize_height)
        elseif resize_mode == 'half_crop' then
            -- resize to average ratio keeping aspect ratio
            new_ratio = (width_ratio + height_ratio) / 2.0
            resize_width = round(img:size(3) / new_ratio)
            resize_height = round(img:size(2) / new_ratio)

            if width_ratio > height_ratio and (height - resize_height) % 2 == 1 then
                resize_height = resize_height + 1
            elseif width_ratio < height_ratio and (width - resize_width) % 2 == 1 then
                resize_width = resize_width + 1
            end

            img = image.scale(img,resize_width,resize_height) 
            -- chop off ends of dimension that is still too long
            if width_ratio > height_ratio then
                start = round((resize_width-width)/2.0)
                img = image.crop(img, start, 0, start+width, img:size(2))
            else
                start = round((resize_height-height)/2.0)
                img = image.crop(img, 0, start, img:size(3), start+height)
            end
        end

        --return img if it reaches the expected size
        if img:size(2) == height and img:size(3) == width then
            return img
        end

        -- fill ends of dimension that is too short with random noise
        if width_ratio > height_ratio then
            padding = (height - resize_height)/2
            padding_tensor = torch.rand(img:size(1),padding,img:size(3))            
	    img = torch.cat(padding_tensor,img,2)
	    img = torch.cat(img,padding_tensor,2)
        else
            padding = (width - resize_width)/2
            padding_tensor = torch.rand(img:size(1),img:size(2),padding)
	    img = torch.cat(padding_tensor,img,3)
	    img = torch.cat(img,padding_tensor,3)
        end
        return img

    end
end

-- This module corrects the final output dimension of the model to match with total number of classes. This follows Reverse Depth First Search approach i.e., modules of the model are checked from last to first, and internally for each module same procedure was followed to find the module with weights. If the module has weights, it indicates the input and output sizes. If the first dimension of weight is not same as total classes count, then corrects the dimension to match with classes count.
-- TODO: currently this module supports only Linear module and for all other modules it will just display a warning message.
function correctFinalOutputDim(node, outputSize)
    if (node.modules ~= nil) then
        if (type(node.modules) == 'table') then
            for i=#node.modules,1,-1 do
                local child = node.modules[i]
                if correctFinalOutputDim(child, outputSize) then
                    return true
                end
            end
        end
    end
    if node.weight then
        if node.weight:size(1) ~= outputSize then
            if torch.type(node) == "nn.Linear" then
		local oldOutputSize = node.weight:size(1)
                node:__init(node.weight:size(2),outputSize)
                logmessage.display(0,'changed output size for ' .. torch.type(node) .. ', from ' .. oldOutputSize .. ' to ' .. outputSize)
            else
                logmessage.display(1,'output size for last ' .. torch.type(node) .. ' layer is ' .. node.weight:size(1) .. ', which is different from total number of classes i.e., ' .. outputSize)
            end
            --logmessage.display(0,model:__tostring__())
        end
        return true
    end
    return false
end

utilsClass.correctFinalOutputDim = correctFinalOutputDim
return utilsClass

