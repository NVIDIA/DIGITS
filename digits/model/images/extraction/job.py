# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path

from digits.utils import subclass, override
from ..job import ImagePretrainedModelJob

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

@subclass
class FeatureExtractionModelJob(ImagePretrainedModelJob):
    """
    A Job that creates an image model for a feature extraction network
    """

    def __init__(self, **kwargs):
        super(FeatureExtractionModelJob, self).__init__(**kwargs)
        self.pickver_job_model_feature_extraction = PICKLE_VERSION

    @override
    def job_type(self):
        return 'Feature Extraction Model'

    @override
    def download_files(self, epoch=-1):
        task = self.load_model_task()

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
                (self.path(task.deploy_file),
                    os.path.basename(task.deploy_file)),
                (snapshot_filename,
                    os.path.basename(snapshot_filename)),
                ]

