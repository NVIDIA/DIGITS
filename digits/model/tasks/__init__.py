# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .caffe_train import CaffeTrainTask
from .torch_train import TorchTrainTask
from .train import TrainTask

__all__ = [
    'CaffeTrainTask',
    'TorchTrainTask',
    'TrainTask',
]
