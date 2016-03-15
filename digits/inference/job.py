# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from . import tasks
import digits.frameworks
from digits.job import Job
from digits.utils import subclass, override

@subclass
class InferenceJob(Job):
    """
    A Job that exercises the forward pass of a neural network
    """

    def __init__(self, model, images, epoch, layers, **kwargs):
        """
        Arguments:
        model   -- job object associated with model to perform inference on
        images  -- list of image paths to perform inference on
        epoch   -- epoch of model snapshot to use
        layers  -- layers to import ('all' or 'none')

        Keyword arguments:
        ground_truths -- desired output
        """

        ground_truths = kwargs.pop('ground_truths', None)

        super(InferenceJob, self).__init__(volatile = True, **kwargs)

        # get handle to framework object
        fw_id = model.train_task().framework_id
        fw = digits.frameworks.get_framework_by_id(fw_id)

        # create inference task
        self.tasks.append(fw.create_inference_task(
            job_dir   = self.dir(),
            model     = model,
            images    = images,
            epoch     = epoch,
            layers        = layers,
            ground_truths = ground_truths))

    @override
    def __getstate__(self):
        fields_to_save = ['_id', '_name']
        full_state = super(InferenceJob, self).__getstate__()
        state_to_save = {}
        for field in fields_to_save:
            state_to_save[field] = full_state[field]
        return state_to_save

    def inference_task(self):
        """Return the first and only InferenceTask for this job"""
        return [t for t in self.tasks if isinstance(t, tasks.InferenceTask)][0]

    @override
    def __setstate__(self, state):
        super(InferenceJob, self).__setstate__(state)

    def get_data(self):
        """Return inference data"""
        task = self.inference_task()
        return task.inference_inputs, task.inference_outputs, task.inference_layers

    def get_parameters(self):
        """Return a tuple of inference parameters: (model, image_list, ground_truths,) """
        task = self.inference_task()
        return task.model, task.images, task.ground_truths

    @override
    def on_status_update(self):
        super(InferenceJob, self).on_status_update()

        from digits.webapp import app, socketio

        if not self.status.is_running():
            message = {
                    'job_id': self.id(),
                    }

            socketio.emit('job reload_page',
                    message,
                    namespace='/jobs',
                    room=self.id(),
                    )


    