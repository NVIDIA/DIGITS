# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

from framework import Framework
from caffe_framework import CaffeFramework
from digits.config import config_value

#
#  create framework instances
#

# caffe is mandatory
caffe = CaffeFramework()

#
#  utility functions
#

def get_frameworks():
    """
    return list of all available framework instances
    there may be more than one instance per framework class
    """
    frameworks = [caffe]
    return frameworks

def get_framework_by_id(framework_id):
    """
    return framework instance associated with given id
    """
    for fw in get_frameworks():
        if fw.get_id() == framework_id:
            return fw
    return None


