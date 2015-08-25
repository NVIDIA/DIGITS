# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

from framework import Framework
from caffe_framework import CaffeFramework
from torch_framework import TorchFramework

#
#  create instances
#  TODO: do this dynamically depending on framework availability
#
torch = TorchFramework()
caffe = CaffeFramework()

# return list of all available framework instances
def get_frameworks():
    return [caffe, torch]

# return framework associated with given id
def get_framework_by_id(id):
    for fw in get_frameworks():
        if fw.get_id() == id:
            return fw
    return None


