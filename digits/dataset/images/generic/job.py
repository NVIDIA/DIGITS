# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import os.path

from digits.dataset import tasks
from digits import utils
from digits.utils import subclass, override
from digits.status import Status
from ..job import ImageDatasetJob

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

@subclass
class GenericImageDatasetJob(ImageDatasetJob):
    """
    A Job that creates an image dataset for a generic network
    """

    def __init__(self, **kwargs):
        self.mean_file = kwargs.pop('mean_file', None)
        super(GenericImageDatasetJob, self).__init__(**kwargs)
        self.pickver_job_dataset_image_generic = PICKLE_VERSION

    def __setstate__(self, state):
        super(GenericImageDatasetJob, self).__setstate__(state)
        self.pickver_job_dataset_image_generic = PICKLE_VERSION

    @override
    def job_type(self):
        return 'Generic Image Dataset'

