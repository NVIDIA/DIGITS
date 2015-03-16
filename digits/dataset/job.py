# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

from digits.job import Job
from . import tasks

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

class DatasetJob(Job):
    """
    A Job that creates a dataset
    """

    def __init__(self, **kwargs):
        """
        """
        super(DatasetJob, self).__init__(**kwargs)
        self.pickver_job_dataset = PICKLE_VERSION

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

    def test_db_task(self):
        """
        Return the task that creates the test set
        """
        for t in self.tasks:
            if isinstance(t, tasks.CreateDbTask) and 'test' in t.name().lower():
                return t
        return None

