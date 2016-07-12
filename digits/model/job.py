# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from os import path

from . import tasks
from digits.job import Job
from digits.utils import override

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

class ModelJob(Job):
    """
    A Job that creates a neural network model
    """

    def __init__(self, dataset_id, **kwargs):
        """
        Arguments:
        dataset_id -- the job_id of the DatasetJob that this ModelJob depends on
        """
        super(ModelJob, self).__init__(**kwargs)
        self.pickver_job_dataset = PICKLE_VERSION

        self.custom_mean_path = None

        self.dataset_id = dataset_id
        self.load_dataset()

    def __getstate__(self):
        state = super(ModelJob, self).__getstate__()
        if 'dataset' in state:
            del state['dataset']
        return state

    def __setstate__(self, state):
        super(ModelJob, self).__setstate__(state)
        self.dataset = None

    @override
    def json_dict(self, verbose=False):
        d = super(ModelJob, self).json_dict(verbose)
        d['dataset_id'] = self.dataset_id

        if verbose:
            d.update({
                'snapshots': [s[1] for s in self.train_task().snapshots],
                })
        return d

    def load_dataset(self):
        from digits.webapp import scheduler
        job = scheduler.get_job(self.dataset_id)
        assert job is not None, 'Cannot find dataset'
        self.dataset = job
        for task in self.tasks:
            task.dataset = job

    def train_task(self):
        """Return the first TrainTask for this job"""
        return [t for t in self.tasks if isinstance(t, tasks.TrainTask)][0]

    def download_files(self):
        """
        Returns a list of tuples: [(path, filename)...]
        These files get added to an archive when this job is downloaded
        """
        return NotImplementedError()

    def set_custom_mean_path(self, custom_mean_path):
        """
        Set the path to a custom mean image protoblob to be used by the model job. If the custom mean path is None, it
        should be assumed that the mean image will be that of the dataset.
        returns: True if the custom_mean_path was successfully set, False otherwise
        """
        if custom_mean_path != '' and path.isfile(custom_mean_path) is True:
            self.custom_mean_path = custom_mean_path
            return True
        else:
            return False

    def get_mean_path(self):
        """
        returns the file path to the mean image protoblob\jpg\file, which is either the dataset's (default) or some
        user specified path
        """

        if self.dataset is None:
            self.load_dataset()

        if self.custom_mean_path is None:
            # The user never set the mean path, so the dataset's mean path should be used
            return self.dataset.path(self.dataset.get_mean_file())
        else:
            # The user has specified the file path to use
            return self.dataset.path(self.custom_mean_path)