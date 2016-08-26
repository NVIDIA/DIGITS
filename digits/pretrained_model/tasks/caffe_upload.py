# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import
import os
import digits
from digits.utils import subclass, override
from digits.status import Status
from digits.pretrained_model.tasks import UploadPretrainedModelTask

@subclass
class CaffeUploadTask(UploadPretrainedModelTask):
    def __init__(self, **kwargs):
        super(CaffeUploadTask, self).__init__(**kwargs)

    @override
    def name(self):
        return 'Upload Pretrained Caffe Model'

    @override
    def get_model_def_path(self):
        """
        Get path to model definition
        """
        return os.path.join(self.job_dir, "original.prototxt")

    @override
    def get_weights_path(self):
        """
        Get path to model weights
        """
        return os.path.join(self.job_dir, "model.caffemodel")


    @override
    def __setstate__(self, state):
        super(CaffeUploadTask, self).__setstate__(state)

    @override
    def run(self, resources):

        self.move_file(self.weights_path, "model.caffemodel")
        self.move_file(self.model_def_path, "original.prototxt")

        if self.labels_path is not None:
            self.move_file(self.labels_path, "labels.txt")

        self.status = Status.DONE
