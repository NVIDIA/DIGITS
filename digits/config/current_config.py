# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

from jobs_dir import JobsDirOption
from gpu_list import GpuListOption
from log_file import LogFileOption
from log_level import LogLevelOption
from server_name import ServerNameOption
from secret_key import SecretKeyOption
from caffe_option import CaffeOption
from torch_option import TorchOption

option_list = None

def reset():
    """
    Reset option_list to a list of unset Options
    """
    global option_list

    option_list = [
            JobsDirOption(),
            GpuListOption(),
            LogFileOption(),
            LogLevelOption(),
            ServerNameOption(),
            SecretKeyOption(),
            CaffeOption(),
            TorchOption(),
            ]

reset()

def config_value(key):
    """
    Return the current configuration value for the given option

    Arguments:
    key -- the key of the configuration option
    """
    for option in option_list:
        if key == option.config_file_key():
            if not option.valid():
                raise RuntimeError('No valid value set for "%s"' % key)
            return option.config_dict_value()
    raise RuntimeError('No option found for "%s"' % key)

