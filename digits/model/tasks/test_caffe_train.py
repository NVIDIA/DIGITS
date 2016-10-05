# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from . import caffe_train
from digits import test_utils

def test_caffe_imports():
    test_utils.skipIfNotFramework('caffe')

    import numpy
    import google.protobuf

