-- Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.

-- This file contains the logic of printing log messgages
local logmessage = torch.class('logmessage')

-------------------------------------------------------------------------------------------------------------
-- display function accepts two input parameters:
-- parameter_name         format           description
-- levelcode              number           specifies the severity of message i.e., 0=info, 1=warn, 2=error.
-- message                string           message to be displayed

-- Usage:
-- require 'logmessage'
-- logmessage.display(0,'This is informational message as the levelcode is 0')
-------------------------------------------------------------------------------------------------------------
function logmessage.display(levelcode, message)
  local levelname=nil
  if levelcode == 0 then
    levelname="INFO "
  elseif levelcode == 1 then
    levelname="WARNING"
  elseif levelcode == 2 then
    levelname="ERROR"
  elseif levelcode == 3 then
    levelname="FAIL"
  end
  print(os.date("%Y-%m-%d %H:%M:%S") .. ' [' .. levelname .. '] ' .. message)
end

