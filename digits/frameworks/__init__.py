# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .caffe_framework import CaffeFramework
from .framework import Framework
from .torch_framework import TorchFramework
from digits.config import config_value

__all__ = [
    'Framework',
    'CaffeFramework',
    'TorchFramework',
]

#
#  create framework instances
#

# torch is optional
torch = TorchFramework() if config_value('torch')['enabled'] else None

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
    if torch:
        frameworks.append(torch)
    return frameworks


def get_framework_by_id(framework_id):
    """
    return framework instance associated with given id
    """
    for fw in get_frameworks():
        if fw.get_id() == framework_id:
            return fw
    return None
