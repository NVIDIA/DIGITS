# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import
import os
import digits
from digits import frameworks
from digits.utils import subclass, override
from digits.status import Status
from digits.pretrained_model.tasks import UploadPretrainedModelTask
from digits.framework_helpers import caffe_helpers

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1


@subclass
class CaffeUploadTask(UploadPretrainedModelTask):
    def __init__(self, **kwargs):
        super(CaffeUploadTask, self).__init__(**kwargs)
        self.pickver_task_caffe_upload = PICKLE_VERSION

        self.model_file = caffe_helpers.CAFFE_ORIGINAL_FILE
        self.deploy_file = caffe_helpers.CAFFE_DEPLOY_FILE
        self.weights_file = caffe_helpers.CAFFE_WEIGHTS_FILE

    @override
    def name(self):
        return 'Upload Pretrained Caffe Model'

    @override
    def get_model_def_path(self):
        """
        Get path to model definition
        """
        return os.path.join(self.job_dir,self.model_file)

    @override
    def get_weights_path(self):
        """
        Get path to model weights
        """
        return os.path.join(self.job_dir,self.weights_file)

    @override
    def get_deploy_path(self):
        """
        Get path to file containing model def for deploy/visualization
        """
        return os.path.join(self.job_dir,self.deploy_file)

    @override
    def write_deploy(self):
        # get handle to framework object
        fw = frameworks.get_framework_by_id("caffe")
        model_def_path  = self.get_model_def_path()
        network = fw.get_network_from_path(model_def_path)

        image_dim = [int(self.image_info["height"]), int(self.image_info["width"]),int(self.image_info["image_type"]) ]

        caffe_helpers.save_deploy_file_classification(network,self.job_dir,len(self.get_labels()),None,image_dim,None)

    @override
    def __setstate__(self, state):
        if 'pickver_task_caffe_upload' not in state:
            state['model_file']   = caffe_helpers.CAFFE_ORIGINAL_FILE
            state['deploy_file']  = caffe_helpers.CAFFE_DEPLOY_FILE
            state['weights_file'] = caffe_helpers.CAFFE_WEIGHTS_FILE

        state['pickver_task_caffe_upload'] = PICKLE_VERSION

        super(CaffeUploadTask, self).__setstate__(state)

    @override
    def run(self, resources):

        self.move_file(self.weights_path, "model.caffemodel")
        self.move_file(self.model_def_path, "original.prototxt")

        if self.labels_path is not None:
            self.move_file(self.labels_path, "labels.txt")

        if self.mean_path is not None:
            self.move_file(self.mean_path, "mean.binaryproto")

        self.write_deploy()
        self.status = Status.DONE
