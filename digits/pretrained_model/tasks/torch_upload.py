# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import
import os
from digits.utils import subclass, override
from digits.status import Status
from digits.pretrained_model.tasks import UploadPretrainedModelTask


@subclass
class TorchUploadTask(UploadPretrainedModelTask):

    def __init__(self, **kwargs):
        super(TorchUploadTask, self).__init__(**kwargs)

    @override
    def name(self):
        return 'Upload Pretrained Torch Model'

    @override
    def get_model_def_path(self):
        """
        Get path to model definition
        """
        return os.path.join(self.job_dir, "original.lua")

    @override
    def get_weights_path(self):
        """
        Get path to model weights
        """
        return os.path.join(self.job_dir, "_Model.t7")

    @override
    def __setstate__(self, state):
        super(TorchUploadTask, self).__setstate__(state)

    @override
    def run(self, resources):

        self.move_file(self.weights_path, "_Model.t7")
        self.move_file(self.model_def_path, "original.lua")

        if self.labels_path is not None:
            self.move_file(self.labels_path, "labels.txt")

        self.status = Status.DONE
