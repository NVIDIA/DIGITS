-- Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.

require 'torch'   -- torch
require 'image'   -- for color transforms

package.path = debug.getinfo(1, "S").source:match[[^@?(.*[\/])[^\/]-$]] .."?.lua;".. package.path

require 'logmessage'

------------- UTILITY FUNCTIONS ----------------------------

local utilsClass={}

-- round function
local function round(num, idp)
  print ('entered')
  local mult = 10^(idp or 0)
  return math.floor(num * mult + 0.5) / mult
end

utilsClass.round = round

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


return utilsClass

