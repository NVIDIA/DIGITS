# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from digits.utils import subclass
from flask.ext.wtf import Form


@subclass
class ConfigForm(Form):
    pass
