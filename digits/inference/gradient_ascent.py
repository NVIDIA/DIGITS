# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import digits.frameworks
from digits.job import Job
from digits.utils import subclass, override
from digits.inference.tasks import GradientAscentTask

@subclass
class GradientAscentJob(Job):
    """
    A Job that exercises getting the max activations for units in pretrained model
    """

    def __init__(self, pretrained_model, layer, units, **kwargs):
        """
        Arguments:
        pretrained_model -- job object associated with pretrained_model
        layer -- layer to get activations for
        units -- array of indices to got max activations for
        """

        super(GradientAscentJob, self).__init__(persistent = False, **kwargs)
        self.pretrained_model = pretrained_model
        # create inference task
        self.tasks.append(GradientAscentTask(
            pretrained_model,
            layer,
            units,
            job_dir = self.dir()
            )
        )

    @override
    def __getstate__(self):
        fields_to_save = ['_id', '_name']
        full_state = super(GradientAscentJob, self).__getstate__()
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

    def inference_task(self):
        """Return the first and only InferenceTask for this job"""
        return [t for t in self.tasks if isinstance(t, tasks.GradientAscentTask)][0]

    @override
    def __setstate__(self, state):
        super(GradientAscentJob, self).__setstate__(state)

    def get_data(self):
        """Return inference data"""
        task = self.inference_task()
