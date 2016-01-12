# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from ..job import ModelJob

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

class ImageModelJob(ModelJob):
    """
    A Job that creates an image model
    """

    def __init__(self, **kwargs):
        """
        """
        super(ImageModelJob, self).__init__(**kwargs)
        self.pickver_job_model_image = PICKLE_VERSION

