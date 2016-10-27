# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

# Create this object before importing the following imports, since they edit the list
option_list = {}

from . import caffe
from . import gpu_list
from . import jobs_dir
from . import log_file
from . import torch
from . import server_name
from . import store_option


def config_value(option):
    """
    Return the current configuration value for the given option
    """
    return option_list[option]
