# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

from digits.job import Job
from digits.utils import subclass, override
from . import tasks

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

@subclass
class DatasetJob(Job):
    """
    A Job that creates a dataset
    """

    def __init__(self, **kwargs):
        """
        """
        super(DatasetJob, self).__init__(**kwargs)
        self.pickver_job_dataset = PICKLE_VERSION

    @override
    def json_dict(self, verbose=False):
        d = super(DatasetJob, self).json_dict(verbose)

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

    def create_db_tasks(self):
        """
        Return all CreateDbTasks for this job
        """
        return [t for t in self.tasks if isinstance(t, tasks.CreateDbTask)]

    def train_db_task(self):
        """
        Return the task that creates the training set
        """
        raise NotImplementedError('Please implement me')

    def val_db_task(self):
        """
        Return the task that creates the validation set
        """
        for t in self.tasks:
            if isinstance(t, tasks.CreateDbTask) and 'val' in t.name().lower():
                return t
        return None

    def test_db_task(self):
        """
        Return the task that creates the test set
        """
        for t in self.tasks:
            if isinstance(t, tasks.CreateDbTask) and 'test' in t.name().lower():
                return t
        return None

    def analyze_db_tasks(self):
        """
        Return all AnalyzeDbTasks for this job
        """
        return [t for t in self.tasks if isinstance(t, tasks.AnalyzeDbTask)]

