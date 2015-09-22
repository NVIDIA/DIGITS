# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

from digits.utils import subclass, override
from ..job import ImageDatasetJob
from digits.dataset import tasks

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

    @override
    def train_db_task(self):
        """
        Return the task that creates the training set
        """
        for t in self.tasks:
            if isinstance(t, tasks.AnalyzeDbTask) and 'train' in t.name().lower():
                return t
        return None


