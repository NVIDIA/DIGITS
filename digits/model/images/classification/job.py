# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path

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

    @override
    def download_files(self, epoch=-1):
        task = self.train_task()

        snapshot_filename = None
        if epoch == -1 and len(task.snapshots):
            epoch = task.snapshots[-1][1]
            snapshot_filename = task.snapshots[-1][0]
        else:
            for f, e in task.snapshots:
                if e == epoch:
                    snapshot_filename = f
                    break
        if not snapshot_filename:
            raise ValueError('Invalid epoch')

        return [
                (self.path(task.get_model_file()),
                    os.path.basename(task.get_model_file())),
                (task.dataset.path(task.dataset.labels_file),
                    os.path.basename(task.dataset.labels_file)),
                (task.dataset.path(task.dataset.train_db_task().mean_file),
                    os.path.basename(task.dataset.train_db_task().mean_file)),
                (snapshot_filename,
                    os.path.basename(snapshot_filename)),
                ]

