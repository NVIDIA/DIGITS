# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import base64
from collections import OrderedDict
import os.path
import shutil
import tempfile
import re
import sys

import h5py

import digits
from digits import device_query
from digits.task import Task
from digits.utils import subclass, override
from digits.status import Status
import subprocess

@subclass
class UploadPretrainedModelTask(Task):
    """
    A task for uploading pretrained models
    """

    def __init__(self, weights_path, model_def_path, labels_path=None, framework="caffe", **kwargs):

        """
        Arguments:
        weights_path -- path to model weights (**.caffemodel or ***.t7)
        model_def_path  -- path to model definition (**.prototxt or ***.lua)
        """
        self.weights_path = weights_path
        self.model_def_path = model_def_path
        self.labels_path = labels_path
        self.framework = framework

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

    def move_file(self,input_file, output):
        shutil.move(input_file, self.job_dir+"/"+output)

    @override
    def run(self, resources):
        env = os.environ.copy()
        if self.framework == "caffe":
            self.move_file(self.weights_path, "model.caffemodel")
            self.move_file(self.model_def_path, "original.prototxt")
        else:
            self.move_file(self.weights_path, "_Model.t7")
            self.move_file(self.model_def_path, "original.lua")

        if self.labels_path is not None:
            self.move_file(self.labels_path, "labels.txt")

        self.status = Status.DONE
