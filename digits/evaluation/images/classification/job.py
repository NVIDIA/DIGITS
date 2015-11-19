# -*- coding: utf-8 -*-

from ..job import ImageEvaluationJob
from digits.utils import subclass, override

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

@subclass
class ImageClassificationEvaluationJob(ImageEvaluationJob):
    """
    A Job that creates a performance analysis for a classification network
    """

    def __init__(self, model_id, model_epoch, **kwargs):
        """
        Arguments:
        model_id    -- the classification model job id
        model_epoch -- the epoch corresponding to the snapshot we want to evaluate

        Keyword arguments:
        """

        super(ImageClassificationEvaluationJob, self).__init__(model_id=model_id, **kwargs)

        self.model_epoch = model_epoch  

        self.pickver_job_evaluation_image_classification = PICKLE_VERSION
        self.labels_file = None

        task = self.model_job.train_task()

        snapshot_filename = None
        if self.model_epoch == -1 and len(task.snapshots):
            self.model_epoch = task.snapshots[-1][1]
            snapshot_filename = task.snapshots[-1][0]
        else:
            for f, e in task.snapshots:
                if e == self.model_epoch:
                    snapshot_filename = f
                    break
        if not snapshot_filename:
            raise ValueError('Invalid epoch')

        self.snapshot_filename = snapshot_filename

    @override
    def job_type(self):
        return 'Image Classification Model Evaluation'

    @override
    def parent_jobs(self):
        return [self.model_job]
