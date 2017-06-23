# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .caffe_train import CaffeTrainTask
from .tensorflow_train import TensorflowTrainTask
from .torch_train import TorchTrainTask
from .train import TrainTask

__all__ = [
    'CaffeTrainTask',
    'TorchTrainTask',
    'TrainTask',
]
