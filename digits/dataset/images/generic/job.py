# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from ..job import ImageDatasetJob
from digits.dataset import tasks
from digits.utils import subclass, override, constants

# NOTE: Increment this every time the pickled object changes
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

    def analyze_db_task(self, stage):
        """
        Return AnalyzeDbTask for this stage
        """
        if stage == constants.TRAIN_DB:
            s = 'train'
        elif stage == constants.VAL_DB:
            s = 'val'
        else:
            return None
        for t in self.tasks:
            if isinstance(t, tasks.AnalyzeDbTask) and s in t.name().lower():
                return t
        return None

    def analyze_db_tasks(self):
        """
        Return all AnalyzeDbTasks for this job
        """
        return [t for t in self.tasks if isinstance(t, tasks.AnalyzeDbTask)]

    @override
    def get_backend(self):
        """
        Return the DB backend used to create this dataset
        """
        return self.analyze_db_task(constants.TRAIN_DB).backend

    @override
    def get_entry_count(self, stage):
        """
        Return the number of entries in the DB matching the specified stage
        """
        task = self.analyze_db_task(stage)
        return task.image_count if task is not None else 0

    @override
    def get_feature_db_path(self, stage):
        """
        Return the absolute feature DB path for the specified stage
        """
        db = None
        if stage == constants.TRAIN_DB:
            s = 'Training'
        elif stage == constants.VAL_DB:
            s = 'Validation'
        else:
            return None
        for task in self.tasks:
            if task.purpose == '%s Images' % s:
                db = task
        return self.path(db.database) if db else None

    @override
    def get_feature_dims(self):
        """
        Return the shape of the feature N-D array
        """
        db_task = self.analyze_db_task(constants.TRAIN_DB)
        return [db_task.image_height, db_task.image_width, db_task.image_channels]

    @override
    def get_label_db_path(self, stage):
        """
        Return the absolute label DB path for the specified stage
        """
        db = None
        if stage == constants.TRAIN_DB:
            s = 'Training'
        elif stage == constants.VAL_DB:
            s = 'Validation'
        else:
            return None
        for task in self.tasks:
            if task.purpose == '%s Labels' % s:
                db = task
        return self.path(db.database) if db else None

    @override
    def get_mean_file(self):
        """
        Return the mean file
        """
        return self.mean_file

    @override
    def job_type(self):
        return 'Generic Image Dataset'
