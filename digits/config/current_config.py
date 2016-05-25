# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .caffe_option import CaffeOption
from .extension_list import DataExtensionListOption, ViewExtensionListOption
from .gpu_list import GpuListOption
from .jobs_dir import JobsDirOption
from .log_file import LogFileOption
from .log_level import LogLevelOption
from .torch_option import TorchOption
from .server_name import ServerNameOption
from .secret_key import SecretKeyOption

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
            DataExtensionListOption(),
            ViewExtensionListOption(),
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

