# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import
import os

from . import tasks
import digits.frameworks
from digits.job import Job
from digits.utils import subclass, override
from digits.pretrained_model.tasks import CaffeUploadTask, TorchUploadTask

@subclass
class PretrainedModelJob(Job):
    """
    A Job that uploads a pretrained model
    """

    def __init__(self, weights_path, model_def_path, labels_path=None,framework="caffe",
                image_type="3",resize_mode="Squash", width=224, height=224, **kwargs):
        super(PretrainedModelJob, self).__init__(persistent = False, **kwargs)

        self.framework  = framework
        self.image_info = {
            "image_type": image_type,
            "resize_mode": resize_mode,
            "width": width,
            "height": height
        }

        self.tasks = []

        taskKwargs = {
            "weights_path": weights_path,
            "model_def_path": model_def_path,
            "image_info": self.image_info,
            "labels_path": labels_path,
            "job_dir": self.dir()
        }

        if self.framework == "caffe":
            self.tasks.append(CaffeUploadTask(**taskKwargs))
        else:
            self.tasks.append(TorchUploadTask(**taskKwargs))

    def get_weights_path(self):
        return self.tasks[0].get_weights_path()

    def get_model_def_path(self):
        return self.tasks[0].get_model_def_path()

    def has_labels_file(self):
        return os.path.isfile(self.tasks[0].get_labels_path())

    @override
    def is_persistent(self):
        return True

    @override
    def job_type(self):
        return "Pretrained Model"

    @override
    def __getstate__(self):
        fields_to_save = ['_id', '_name', 'username', 'tasks', 'status_history', 'framework', 'image_info']
        full_state = super(PretrainedModelJob, self).__getstate__()
        state_to_save = {}
        for field in fields_to_save:
            state_to_save[field] = full_state[field]
        return state_to_save

    @override
    def __setstate__(self, state):
        super(PretrainedModelJob, self).__setstate__(state)
