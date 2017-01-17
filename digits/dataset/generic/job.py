# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from ..job import DatasetJob
from digits.dataset import tasks
from digits.utils import subclass, override, constants

# NOTE: Increment this every time the pickled object changes
PICKLE_VERSION = 1


@subclass
class GenericDatasetJob(DatasetJob):
    """
    A Job that creates a dataset using a user-defined extension
    """

    def __init__(self,
                 backend,
                 feature_encoding,
                 label_encoding,
                 batch_size,
                 num_threads,
                 force_same_shape,
                 extension_id,
                 extension_userdata,
                 **kwargs
                 ):
        self.backend = backend
        self.feature_encoding = feature_encoding
        self.label_encoding = label_encoding
        self.num_threads = num_threads
        self.force_same_shape = force_same_shape
        self.batch_size = batch_size
        self.extension_id = extension_id
        self.extension_userdata = extension_userdata

        super(GenericDatasetJob, self).__init__(**kwargs)
        self.pickver_job_dataset_extension = PICKLE_VERSION

        # create tasks
        for stage in [constants.TRAIN_DB, constants.VAL_DB, constants.TEST_DB]:
            self.tasks.append(tasks.CreateGenericDbTask(
                job_dir=self.dir(),
                job=self,
                backend=self.backend,
                stage=stage,
            )
            )

    def __setstate__(self, state):
        super(GenericDatasetJob, self).__setstate__(state)
        self.pickver_job_dataset_extension = PICKLE_VERSION

    def create_db_task(self, stage):
        for t in self.tasks:
            if t.stage == stage:
                return t
        return None

    def create_db_tasks(self):
        return self.tasks

    @override
    def get_backend(self):
        """
        Return the DB backend used to create this dataset
        """
        return self.backend

    @override
    def get_entry_count(self, stage):
        """
        Return the number of entries in the DB matching the specified stage
        """
        return self.create_db_task(stage).entry_count

    @override
    def get_feature_db_path(self, stage):
        """
        Return the absolute feature DB path for the specified stage
        """
        return self.path(self.create_db_task(stage).dbs['features'])

    @override
    def get_feature_dims(self):
        """
        Return the shape of the feature N-D array
        """
        shape = self.create_db_task(constants.TRAIN_DB).feature_shape
        if len(shape) == 3:
            # assume image and convert CHW => HWC (numpy default for images)
            shape = [shape[1], shape[2], shape[0]]
        return shape

    @override
    def get_label_db_path(self, stage):
        """
        Return the absolute label DB path for the specified stage
        """
        return self.path(self.create_db_task(stage).dbs['labels'])

    @override
    def get_mean_file(self):
        """
        Return the mean file (if it exists, or None)
        """
        mean_file = self.create_db_task(constants.TRAIN_DB).mean_file
        return self.path(mean_file) if mean_file else ''

    @override
    def job_type(self):
        return 'Generic Dataset'

    @override
    def json_dict(self, verbose=False):
        d = super(GenericDatasetJob, self).json_dict(verbose)
        if verbose:
            d.update({
                'create_db_tasks': [{
                    "name": t.name(),
                    "stage": t.stage,
                    "entry_count": t.entry_count,
                    "feature_db_path": t.dbs['features'],
                    "label_db_path": t.dbs['labels'],
                } for t in self.create_db_tasks()],
                'feature_dims': self.get_feature_dims()})
        return d
