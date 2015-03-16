# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path

from digits.model import tasks
from digits import utils
from digits.utils import subclass, override
from ..job import ImageModelJob

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

@subclass
class ImageClassificationModelJob(ImageModelJob):
    """
    A Job that creates an image model for a classification network
    """

    def __init__(self, **kwargs):
        super(ImageClassificationModelJob, self).__init__(**kwargs)
        self.pickver_job_model_image_classification = PICKLE_VERSION

    @override
    def job_type(self):
        return 'Image Classification Model'

