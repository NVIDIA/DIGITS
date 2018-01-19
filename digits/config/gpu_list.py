# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from . import option_list
import digits.device_query

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

option_list['gpu_list'] = ','.join([str(x) for x in xrange(len(digits.device_query.get_devices()))])
