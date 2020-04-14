# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .framework import Framework
from digits.config import config_value

__all__ = [
]

caffe = None
tensorflow = None

if config_value('caffe')['loaded']:
    from .caffe_framework import CaffeFramework
    caffe = CaffeFramework()
    __all__.append('CaffeFramework')

if config_value('tensorflow')['enabled']:
    from .tensorflow_framework import TensorflowFramework
    tensorflow = TensorflowFramework()
    __all__.append('TensorflowFramework')

if len(__all__) == 0:
    __all__.append('Framework')
#
#  utility functions
#


def get_frameworks():
    """
    return list of all available framework instances
    there may be more than one instance per framework class
    """
    frameworks = []
    if caffe:
        frameworks.append(caffe)
    if tensorflow:
        frameworks.append(tensorflow)
    if len(frameworks) == 0:
        frameworks.append(Framework())
    return frameworks


def get_framework_by_id(framework_id):
    """
    return framework instance associated with given id
    """
    for fw in get_frameworks():
        if fw.get_id() == framework_id:
            return fw
    return None
