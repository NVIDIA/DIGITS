# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

# These are the only two functions that the rest of DIGITS needs to use
from .current_config import config_value
from .load import load_config

if 'DIGITS_MODE_TEST' in os.environ:
    # load the config automatically during testing
    # it's hard to do it manually with nosetests
    load_config()

