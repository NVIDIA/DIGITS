# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import
import os
import subprocess
import digits
from digits.utils import subclass, override
from digits.status import Status
from digits.pretrained_model.tasks import UploadPretrainedModelTask

from digits.config import config_value
from digits import frameworks

@subclass
class TorchUploadTask(UploadPretrainedModelTask):
    def __init__(self, **kwargs):
        super(TorchUploadTask, self).__init__(**kwargs)

    @override
    def name(self):
        return 'Upload Pretrained Torch Model'

    @override
    def get_model_def_path(self,as_json=False):
        """
        Get path to model definition
        """
        if as_json == True:
            return self.job_dir+"/model_def.json"
        else:
            return self.get_deploy_path();

    @override
    def get_weights_path(self):
        """
        Get path to model weights
        """
        return self.job_dir+"/_Model.t7"

    @override
    def get_deploy_path(self):
        """
        Get path to file containing model def for deploy/visualization
        """
        return self.job_dir+"/original.lua"

    @override
    def write_deploy(self,env):
        # Write torch layers to json for layerwise graph visualization
        if config_value('torch_root') == '<PATHS>':
            torch_bin = 'th'
        else:
            torch_bin = os.path.join(config_value('torch_root'), 'bin', 'th')

        args = [torch_bin,
                os.path.join(os.path.dirname(os.path.dirname(digits.__file__)),'tools','torch','toJSON.lua'),
                '--network=%s' % os.path.split(self.get_deploy_path())[1].split(".")[0],
                '--output=%s' % self.get_model_def_path(True),
                ]
        p = subprocess.Popen(args,cwd=self.job_dir,env=env)

    @override
    def __setstate__(self, state):
        super(TorchUploadTask, self).__setstate__(state)

    @override
    def run(self, resources):
        env = os.environ.copy()

        self.move_file(self.weights_path, "_Model.t7",env)
        self.move_file(self.model_def_path, "original.lua",env)

        if self.labels_path is not None:
            self.move_file(self.labels_path, "labels.txt",env)

        self.write_deploy(env)
        self.status = Status.DONE
