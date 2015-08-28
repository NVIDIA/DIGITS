# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

from framework import Framework
from caffe_framework import CaffeFramework
from torch_framework import TorchFramework
from digits.config import config_value

#
#  create framework instances
#

# torch is optional
torch = TorchFramework() if config_value('torch_root') else None

# caffe is mandatory
caffe = CaffeFramework()

# return list of all available framework instances
def get_frameworks():
    frameworks = [caffe]
    if torch:
        frameworks.append(torch)
    return frameworks

# return framework associated with given id
def get_framework_by_id(id):
    for fw in get_frameworks():
        if fw.get_id() == id:
            return fw
    return None


