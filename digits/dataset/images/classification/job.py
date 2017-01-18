# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

from ..job import ImageDatasetJob
from digits.dataset import tasks
from digits.status import Status
from digits.utils import subclass, override, constants

# NOTE: Increment this every time the pickled object changes
PICKLE_VERSION = 2


@subclass
class ImageClassificationDatasetJob(ImageDatasetJob):
    """
    A Job that creates an image dataset for a classification network
    """

    def __init__(self, **kwargs):
        super(ImageClassificationDatasetJob, self).__init__(**kwargs)
        self.pickver_job_dataset_image_classification = PICKLE_VERSION

        self.labels_file = None

    def __setstate__(self, state):
        super(ImageClassificationDatasetJob, self).__setstate__(state)

        if self.pickver_job_dataset_image_classification <= 1:
            task = self.train_db_task()
            if task.image_dims[2] == 3:
                if task.encoding == "jpg":
                    if task.mean_file.endswith('.binaryproto'):
                        import numpy as np
                        import caffe_pb2

                        old_blob = caffe_pb2.BlobProto()
                        with open(task.path(task.mean_file), 'rb') as infile:
                            old_blob.ParseFromString(infile.read())
                        data = np.array(old_blob.data).reshape(
                            old_blob.channels,
                            old_blob.height,
                            old_blob.width)
                        data = data[[2, 1, 0], ...]  # channel swap
                        new_blob = caffe_pb2.BlobProto()
                        new_blob.num = 1
                        new_blob.channels, new_blob.height, new_blob.width = data.shape
                        new_blob.data.extend(data.astype(float).flat)
                        with open(task.path(task.mean_file), 'wb') as outfile:
                            outfile.write(new_blob.SerializeToString())
                else:
                    self.status = Status.ERROR
                    for task in self.tasks:
                        task.status = Status.ERROR
                        task.exception = ('This dataset was created with unencoded '
                                          'RGB channels. Caffe requires BGR input.')

        self.pickver_job_dataset_image_classification = PICKLE_VERSION

    def create_db_tasks(self):
        """
        Return all CreateDbTasks for this job
        """
        return [t for t in self.tasks if isinstance(t, tasks.CreateDbTask)]

    @override
    def get_backend(self):
        """
        Return the DB backend used to create this dataset
        """
        return self.train_db_task().backend

    def get_encoding(self):
        """
        Return the DB encoding used to create this dataset
        """
        return self.train_db_task().encoding

    def get_compression(self):
        """
        Return the DB compression used to create this dataset
        """
        return self.train_db_task().compression

    @override
    def get_entry_count(self, stage):
        """
        Return the number of entries in the DB matching the specified stage
        """
        if stage == constants.TRAIN_DB:
            db = self.train_db_task()
        elif stage == constants.VAL_DB:
            db = self.val_db_task()
        elif stage == constants.TEST_DB:
            db = self.test_db_task()
        else:
            return 0
        return db.entries_count if db is not None else 0

    @override
    def get_feature_dims(self):
        """
        Return the shape of the feature N-D array
        """
        return self.image_dims

    @override
    def get_feature_db_path(self, stage):
        """
        Return the absolute feature DB path for the specified stage
        """
        path = self.path(stage)
        return path if os.path.exists(path) else None

    @override
    def get_label_db_path(self, stage):
        """
        Return the absolute label DB path for the specified stage
        """
        # classification datasets don't have label DBs
        return None

    @override
    def get_mean_file(self):
        """
        Return the mean file
        """
        return self.train_db_task().mean_file

    @override
    def job_type(self):
        return 'Image Classification Dataset'

    @override
    def json_dict(self, verbose=False):
        d = super(ImageClassificationDatasetJob, self).json_dict(verbose)

        if verbose:
            d.update({
                'ParseFolderTasks': [{
                    "name":        t.name(),
                    "label_count": t.label_count,
                    "train_count": t.train_count,
                    "val_count":   t.val_count,
                    "test_count":  t.test_count,
                } for t in self.parse_folder_tasks()],
                'CreateDbTasks': [{
                    "name":             t.name(),
                    "entries":          t.entries_count,
                    "image_width":      t.image_dims[0],
                    "image_height":     t.image_dims[1],
                    "image_channels":   t.image_dims[2],
                    "backend":          t.backend,
                    "encoding":         t.encoding,
                    "compression":      t.compression,
                } for t in self.create_db_tasks()],
            })
        return d

    def parse_folder_tasks(self):
        """
        Return all ParseFolderTasks for this job
        """
        return [t for t in self.tasks if isinstance(t, tasks.ParseFolderTask)]

    def test_db_task(self):
        """
        Return the task that creates the test set
        """
        for t in self.tasks:
            if isinstance(t, tasks.CreateDbTask) and 'test' in t.name().lower():
                return t
        return None

    def train_db_task(self):
        """
        Return the task that creates the training set
        """
        for t in self.tasks:
            if isinstance(t, tasks.CreateDbTask) and 'train' in t.name().lower():
                return t
        return None

    def val_db_task(self):
        """
        Return the task that creates the validation set
        """
        for t in self.tasks:
            if isinstance(t, tasks.CreateDbTask) and 'val' in t.name().lower():
                return t
        return None
