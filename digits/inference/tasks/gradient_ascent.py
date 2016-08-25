# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import base64
from collections import OrderedDict
import h5py
import os
import tempfile
import re
import sys

import digits
from digits import device_query
from digits.task import Task
from digits.utils import subclass, override
from digits.utils.image import embed_image_html
from digits.status import Status
from digits.inference.errors import InferenceError
# from digits.pretrained_model import PretrainedModelJob

@subclass
class GradientAscentTask(Task):
    """
    Get Max Activations from a Pretrained Model
    """

    def __init__(self, pretrained_model, layer, units, **kwargs):
        """
        Arguments:
        pretrained_model -- job relating to a pretrained model
        layer -- name of layer to get activations for
        units -- array of inidices of units to get activations for
        """

        # memorize parameters
        self.weights_path = pretrained_model.get_weights_path()
        self.deploy_path  = pretrained_model.get_deploy_path()

        self.pretrained_model = pretrained_model
        self.layer = layer
        self.units = units
        self.inference_log_file = "inference.log"

        # resources
        self.gpu = None

        # generated data
        self.inference_data_filename = None
        self.inference_inputs = None
        self.inference_outputs = None
        self.inference_layers = []

        super(GradientAscentTask, self).__init__(**kwargs)

    @override
    def name(self):
        return 'Get Max Activations From Pretrained Model'

    @override
    def __getstate__(self):
        state = super(GradientAscentTask, self).__getstate__()
        if 'inference_log' in state:
            # don't save file handle
            del state['inference_log']
        return state

    @override
    def __setstate__(self, state):
        super(GradientAscentTask, self).__setstate__(state)

    @override
    def before_run(self):
        super(GradientAscentTask, self).before_run()
        # create log file
        self.inference_log = open(self.path(self.inference_log_file), 'a')

    @override
    def process_output(self, line):
        self.inference_log.write('%s\n' % line)
        self.inference_log.flush()

        timestamp, level, message = self.preprocess_output_digits(line)
        if not message:
            return False

        # error
        match = re.match(r'Error: (\w+)', message)
        if match:
            message = message.replace('Error: ','')
            from digits.webapp import socketio
            task_info = {
                'task': self.html_id(),
                'update': 'gradient_ascent',
                'data': {
                    'layer': self.layer,
                    'error': message,
                    'id': self.job_id
                    }
                }
            # Update Layer Vis tool:
            socketio.emit('task error',
                    task_info,
                    namespace='/jobs',
                    room=self.pretrained_model.id(),
                    )


        # progress
        match = re.match(r'Processed (\d+)\/(\d+)', message)
        if match:
            self.progress = float(match.group(1))/int(match.group(2))

            from digits.webapp import socketio
            task_info = {
                'task': self.html_id(),
                'update': 'gradient_ascent',
                'data': {
                    'layer': self.layer,
                    'unit': int(match.group(1)),
                    'progress': self.progress,
                    'id': self.job_id
                    },
                'job_id': self.job_id,
                'percentage': int(self.progress*100)
                }

            # Update Layer Vis tool:
            socketio.emit('task update',
                    task_info,
                    namespace='/jobs',
                    room=self.pretrained_model.id(),
                    )

            # Update Job Board:
            task_info['update'] = 'progress'
            socketio.emit('job update',
                    task_info,
                    namespace='/jobs',
                    room="job_management",
                    )

            # Update Satus:

            return True

        # completion
        match = re.match(r'Saved data to (.*)', message)
        if match:
            self.inference_data_filename = match.group(1).strip()
            return True

        return False

    @override
    def after_run(self):
        super(GradientAscentTask, self).after_run()
        self.inference_log.close()

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

    @override
    def task_arguments(self, resources, env):

        if self.pretrained_model.framework == "caffe":
            inference_tool_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(digits.__file__))), 'tools', 'gradient_ascent_caffe.py')
        else:
            inference_tool_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(digits.__file__))), 'tools', 'gradient_ascent_torch.py')

        args = [sys.executable,
            inference_tool_path
            ]

        args.append('--output_dir=%s' % self.pretrained_model.dir())
        args.append('--model_def_path=%s' % self.deploy_path)
        args.append('--weights_path=%s' % self.weights_path)
        args.append('--layer=%s' % self.layer)
        args.append('--units=%s' % ','.join(map(str,self.units)))

        if self.pretrained_model.framework != "caffe":
            args.append('--height=%s' % self.pretrained_model.image_info["height"])
            args.append('--width=%s' % self.pretrained_model.image_info["width"])

        if os.path.isfile(self.pretrained_model.get_mean_path()):
            args.append('--mean_file_path=%s' % self.pretrained_model.get_mean_path())

        if self.gpu is not None:
            args.append('--gpu=%d' % self.gpu)

        return args
