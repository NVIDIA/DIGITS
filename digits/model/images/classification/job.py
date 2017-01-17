# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os.path

from ..job import ImageModelJob
from digits.utils import subclass, override

# NOTE: Increment this every time the pickled object changes
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

        snapshot_filename = task.get_snapshot(epoch)

        # get model files
        model_files = task.get_model_files()
        download_files = [(self.path(filename), os.path.basename(filename))
                          for filename in model_files.values()]

        # add other files
        download_files.extend([
            (task.dataset.path(task.dataset.labels_file),
             os.path.basename(task.dataset.labels_file)),
            (task.dataset.path(task.dataset.get_mean_file()),
             os.path.basename(task.dataset.get_mean_file())),
            (snapshot_filename,
             os.path.basename(snapshot_filename)),
        ])

        return download_files
