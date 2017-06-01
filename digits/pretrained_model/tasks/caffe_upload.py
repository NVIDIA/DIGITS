# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import
import os
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
        tmp_dir = os.path.dirname(self.weights_path)
        python_layer_file_name = 'digits_python_layers.py'
        if os.path.exists(os.path.join(tmp_dir, python_layer_file_name)):
            self.move_file(os.path.join(tmp_dir, python_layer_file_name), python_layer_file_name)
        elif os.path.exists(os.path.join(tmp_dir, python_layer_file_name + 'c')):
            self.move_file(os.path.join(tmp_dir, python_layer_file_name + 'c'), python_layer_file_name + 'c')

        self.status = Status.DONE
