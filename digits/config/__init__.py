# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import os

# These are the only two functions that the rest of DIGITS needs to use
from load import load_config
from current_config import config_value

if 'DIGITS_MODE_TEST' in os.environ:
    # load the config automatically during testing
    # it's hard to do it manually with nosetests
    load_config()

