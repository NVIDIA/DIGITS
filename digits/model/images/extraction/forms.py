# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import wtforms
from wtforms import validators

from ..forms import DummyImageModelForm

class DummyFeatureExtractionModelForm(DummyImageModelForm):
    """
    Defines the form used to create a new DummyFeatureExtractionModelJob
    """
    pass

