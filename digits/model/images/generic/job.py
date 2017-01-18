# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os.path

from ..job import ImageModelJob
from digits.utils import subclass, override

# NOTE: Increment this every time the pickled object changes
PICKLE_VERSION = 1


@subclass
class GenericImageModelJob(ImageModelJob):
    """
    A Job that creates an image model for a generic network
    """

    def __init__(self, **kwargs):
        super(GenericImageModelJob, self).__init__(**kwargs)
        self.pickver_job_model_image_generic = PICKLE_VERSION

    @override
    def job_type(self):
        return 'Generic Image Model'

    @override
    def download_files(self, epoch=-1):
        task = self.train_task()

        snapshot_filename = task.get_snapshot(epoch)

        # get model files
        model_files = task.get_model_files()
        download_files = [(self.path(filename), os.path.basename(filename))
                          for filename in model_files.values()]

        if task.dataset.get_mean_file():
            download_files.append((
                task.dataset.path(task.dataset.get_mean_file()),
                os.path.basename(task.dataset.get_mean_file())))

        # add snapshot
        download_files.append((snapshot_filename,
                               os.path.basename(snapshot_filename)))

        return download_files
