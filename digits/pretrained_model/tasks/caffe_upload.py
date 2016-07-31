# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import
import os
import digits
from digits.utils import subclass, override
from digits.status import Status
from digits.pretrained_model.tasks import UploadPretrainedModelTask

from digits import frameworks
from digits.framework_helpers import caffe_helpers

@subclass
class CaffeUploadTask(UploadPretrainedModelTask):
    def __init__(self, **kwargs):
        super(CaffeUploadTask, self).__init__(**kwargs)

    @override
    def name(self):
        return 'Upload Pretrained Caffe Model'

    @override
    def get_model_def_path(self,as_json=False):
        """
        Get path to model definition
        """
        return self.job_dir+"/original.prototxt"

    @override
    def get_weights_path(self):
        """
        Get path to model weights
        """
        return self.job_dir+"/model.caffemodel"

    @override
    def get_deploy_path(self):
        """
        Get path to file containing model def for deploy/visualization
        """
        return self.job_dir+"/deploy.prototxt"

    @override
    def write_deploy(self,env):
        # get handle to framework object
        fw = frameworks.get_framework_by_id("caffe")
        model_def_path  = self.get_model_def_path()
        network = fw.get_network_from_path(model_def_path)

        image_dim = [ int(self.image_info["width"]), int(self.image_info["height"]), int(self.image_info["image_type"]) ]

        caffe_helpers.save_deploy_file_classification(network,self.job_dir,len(self.get_labels()),None,image_dim,None)

    @override
    def __setstate__(self, state):
        super(CaffeUploadTask, self).__setstate__(state)

    @override
    def run(self, resources):
        env = os.environ.copy()

        self.move_file(self.weights_path, "model.caffemodel",env)
        self.move_file(self.model_def_path, "original.prototxt",env)

        if self.labels_path is not None:
            self.move_file(self.labels_path, "labels.txt",env)

        self.write_deploy(env)
        self.status = Status.DONE
