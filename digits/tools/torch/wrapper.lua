-- Copyright (c) 2015-2017, NVIDIA CORPORATION. All rights reserved.

-- retrieve path to this script so we can import and invoke scripts that
-- are in the same directory
local path = debug.getinfo(1,"S").source
local dir_path = path:match[[^@?(.*[\/])[^\/]-$]]
assert(dir_path ~= nil)
package.path = dir_path .."?.lua;".. package.path

require 'logmessage'

-- custom error handler prints error using logmessage API
function err (x)
    logmessage.display(3, x)
    print(debug.traceback('DIGITS Lua Error',2))
    return "DIGITS error"
end -- err

if #arg < 1 then
    logmessage.display(3, 'Usage: ' .. path .. ' script.lua [args]')
    os.exit(1)
end

-- invoke script within xpcall() to catch errors
if xpcall(function() dofile(dir_path .. arg[1]) end, err) then
    -- OK
    os.exit(0)
else
    -- an error occurred
    os.exit(1)
end


