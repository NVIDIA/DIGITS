# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from digits import test_utils


def test_caffe_imports():
    test_utils.skipIfNotFramework('caffe')

    import numpy  # noqa
    import google.protobuf  # noqa
