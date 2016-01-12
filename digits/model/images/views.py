# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from digits.webapp import app
from .classification import views as _
from .generic import views as _

NAMESPACE = '/models/images'

