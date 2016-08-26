# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import
import os
import subprocess
import digits
from digits.utils import subclass, override
from digits.status import Status
from digits.pretrained_model.tasks import UploadPretrainedModelTask
from digits.config import config_value

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

from digits.config import config_value
from digits import frameworks

@subclass
class TorchUploadTask(UploadPretrainedModelTask):
    def __init__(self, **kwargs):
        super(TorchUploadTask, self).__init__(**kwargs)
        self.pickver_task_torch_upload = PICKLE_VERSION

    @override
    def name(self):
        return 'Upload Pretrained Torch Model'

    @override
    def get_model_def_path(self):
        """
        Get path to model definition
        """
        return self.get_deploy_path()

    @override
    def get_weights_path(self):
        """
        Get path to model weights
        """
        return os.path.join(self.job_dir,"_Model.t7")

    @override
    def get_deploy_path(self):
        """
        Get path to file containing model def for deploy/visualization
        """
        return os.path.join(self.job_dir,"original.lua")

    @override
    def __setstate__(self, state):
        super(TorchUploadTask, self).__setstate__(state)

    @override
    def run(self, resources):

        self.move_file(self.weights_path, "_Model.t7")
        self.move_file(self.model_def_path, "original.lua")

        if self.labels_path is not None:
            self.move_file(self.labels_path, "labels.txt")

        if self.mean_path is not None:
            self.move_file(self.mean_path, "mean.binaryproto")


        self.status = Status.DONE
