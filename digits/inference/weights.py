# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import digits.frameworks
from digits.job import Job
from digits.utils import subclass, override
from digits.inference.tasks import WeightsTask

@subclass
class WeightsJob(Job):
    """
    A Job that exercises getting the weights from a neural network
    """

    def __init__(self, pretrained_model, **kwargs):
        """
        Arguments:
        pretrained_model -- job object associated with pretrained_model to perform inference on
        """

        super(WeightsJob, self).__init__(persistent = False, **kwargs)
        self.pretrained_model = pretrained_model

        # create inference task
        self.tasks.append(WeightsTask(
            pretrained_model,
            job_dir = self.dir()
            )
        )

    @override
    def __getstate__(self):
        fields_to_save = ['_id', '_name']
        full_state = super(WeightsJob, self).__getstate__()
        state_to_save = {}
        for field in fields_to_save:
            state_to_save[field] = full_state[field]
        return state_to_save

    @override
    def delete_timeout(self):
        return 0

    @override
    def is_persistent(self):
        return False

    def weights_task(self):
        """Return the first and only InferenceTask for this job"""
        return [t for t in self.tasks if isinstance(t, tasks.WeightsTask)][0]

    @override
    def __setstate__(self, state):
        super(WeightsJob, self).__setstate__(state)

    def get_data(self):
        """Return inference data"""
        task = self.inference_task()
