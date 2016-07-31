# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import
import subprocess

import digits
from digits.task import Task
from digits.utils import subclass, override


@subclass
class UploadPretrainedModelTask(Task):
    """
    A task for uploading pretrained models
    """
    def __init__(self, **kwargs):
        """
        Arguments:
        weights_path -- path to model weights (**.caffemodel or ***.t7)
        model_def_path  -- path to model definition (**.prototxt or ***.lua)
        image_info -- a dictionary containing image_type, resize_mode, width, and height
        labels_path -- path to text file containing list of labels
        framework  -- framework of this job (ie caffe or torch)
        """
        self.weights_path = kwargs.pop('weights_path', None)
        self.model_def_path = kwargs.pop('model_def_path', None)
        self.image_info = kwargs.pop('image_info', None)
        self.labels_path = kwargs.pop('labels_path', None)
        self.framework = kwargs.pop('framework', None)

        # resources
        self.gpu = None

        super(UploadPretrainedModelTask, self).__init__(**kwargs)

    @override
    def name(self):
        return 'Upload Pretrained Model'

    @override
    def __setstate__(self, state):
        super(UploadPretrainedModelTask, self).__setstate__(state)

    @override
    def process_output(self, line):
        return True

    @override
    def offer_resources(self, resources):
        reserved_resources = {}
        # we need one CPU resource from inference_task_pool
        cpu_key = 'inference_task_pool'
        if cpu_key not in resources:
            return None
        for resource in resources[cpu_key]:
            if resource.remaining() >= 1:
                reserved_resources[cpu_key] = [(resource.identifier, 1)]
                # we reserve the first available GPU, if there are any
                gpu_key = 'gpus'
                if resources[gpu_key]:
                    for resource in resources[gpu_key]:
                        if resource.remaining() >= 1:
                            self.gpu = int(resource.identifier)
                            reserved_resources[gpu_key] = [(resource.identifier, 1)]
                            break
                return reserved_resources
        return None

    def get_labels(self):
        labels = []
        if self.labels_path is not None:
            with open(self.job_dir+"/labels.txt") as f:
                labels = f.readlines()
        return labels

    def move_file(self,input, output,env):
        args  = ["cp", input, self.job_dir+"/"+output]
        p = subprocess.Popen(args,env=env)

    def get_model_def_path(self,as_json=False):
        """
        Get path to model definition
        """
        raise NotImplementedError('Please implement me')

    def get_weights_path(self):
        """
        Get path to model weights
        """
        raise NotImplementedError('Please implement me')

    def get_deploy_path(self):
        """
        Get path to file containing model def for deploy/visualization
        """
        raise NotImplementedError('Please implement me')

    def write_deploy(self):
        """
        Write model definition for deploy/visualization
        """
        raise NotImplementedError('Please implement me')
