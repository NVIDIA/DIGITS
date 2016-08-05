# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from . import tasks
import digits.frameworks
from digits.job import Job
from digits.utils import subclass, override
from digits.pretrained_model.tasks import UploadPretrainedModelTask

@subclass
class PretrainedModelJob(Job):
    """
    A Job that uploads a pretrained model
    """

    def __init__(self, weights_path, model_def_path, labels_path=None,framework="caffe",**kwargs):
        super(PretrainedModelJob, self).__init__(persistent = False, **kwargs)

        self.has_labels = labels_path is not None
        self.framework  = framework
        self.tasks = []
        self.tasks.append(UploadPretrainedModelTask(
            weights_path,
            model_def_path,
            labels_path,
            framework,
            job_dir=self.dir()
        ))

    def get_weights_path(self):
        if self.framework == "caffe":
            return self.dir()+"/model.caffemodel"
        else:
            return self.dir()+"/_Model.t7"

    def get_model_def_path(self):
        if self.framework == "caffe":
            return self.dir()+"/original.prototxt"
        else:
            return self.dir()+"/original.lua"

    @override
    def job_type(self):
        return "Pretrained Model"

    @override
    def __getstate__(self):
        fields_to_save = ['_id', '_name', 'username', 'tasks', 'status_history', 'has_labels', 'framework']
        full_state = super(PretrainedModelJob, self).__getstate__()
        state_to_save = {}
        for field in fields_to_save:
            state_to_save[field] = full_state[field]
        return state_to_save

    @override
    def __setstate__(self, state):
        super(PretrainedModelJob, self).__setstate__(state)
