# -*- coding: utf-8 -*-

from digits.job import Job
from . import tasks
from digits.utils import override

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

class EvaluationJob(Job):
    """
    A Job that performs a performance evaluation
    """

    def __init__(self, model_id, **kwargs):
        """
        """
        super(EvaluationJob, self).__init__(**kwargs)
        self.pickver_job_evaluation = PICKLE_VERSION
        self.model_id = model_id
        self.load_model()


    def accuracy_tasks(self):
        """Return all the Accuracy Tasks for this job"""
        return [t for t in self.tasks if isinstance(t, tasks.AccuracyTask)]

    @override
    def json_dict(self, verbose=False):
        d = super(EvaluationJob, self).json_dict(verbose)
        d['model_id'] = self.model_id
        # if verbose:
        #    TODO
        return d

    def load_model(self):
        """
        Load the ModelJob corresponding to self.model_id
        """
        from digits.webapp import scheduler
        job = scheduler.get_job(self.model_id)
        assert job is not None, 'Cannot find model'
        self.model_job = job

        