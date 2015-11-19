# -*- coding: utf-8 -*-

from ..job import EvaluationJob

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

class ImageEvaluationJob(EvaluationJob):
    """
    A Job that creates a classification performance evaluation
    """

    def __init__(self, **kwargs):
        """
        Arguments:
        """

        super(ImageEvaluationJob, self).__init__(**kwargs)
        self.pickver_job_evaluation_image = PICKLE_VERSION
