# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import re
import subprocess
import time
import tempfile

import flask

from .errors import NetworkVisualizationError
from .framework import Framework
import digits
from digits import utils
from digits.config import config_value
from digits.model.tasks import TorchTrainTask
from digits.utils import subclass, override


@subclass
class TorchFramework(Framework):

    """
    Defines required methods to interact with the Torch framework
    """

    # short descriptive name
    NAME = 'Torch'

    # identifier of framework class
    CLASS = 'torch'

    # whether this framework can shuffle data during training
    CAN_SHUFFLE_DATA = True

    SUPPORTED_SOLVER_TYPES = ['SGD', 'NESTEROV', 'ADAGRAD',
                              'RMSPROP', 'ADADELTA', 'ADAM']

    SUPPORTED_DATA_TRANSFORMATION_TYPES = ['MEAN_SUBTRACTION', 'CROPPING']
    SUPPORTED_DATA_AUGMENTATION_TYPES = ['FLIPPING', 'QUAD_ROTATION', 'ARBITRARY_ROTATION',
                                         'SCALING', 'NOISE', 'HSV_SHIFTING']

    def __init__(self):
        super(TorchFramework, self).__init__()
        # id must be unique
        self.framework_id = self.CLASS

    @override
    def create_train_task(self, **kwargs):
        """
        create train task
        """
        return TorchTrainTask(framework_id=self.framework_id, **kwargs)

    @override
    def get_standard_network_desc(self, network):
        """
        return description of standard network
        """
        networks_dir = os.path.join(os.path.dirname(digits.__file__), 'standard-networks', self.CLASS)

        # Torch's GoogLeNet and AlexNet models are placed in sub folder
        if (network == "alexnet" or network == "googlenet"):
            networks_dir = os.path.join(networks_dir, 'ImageNet-Training')

        for filename in os.listdir(networks_dir):
            path = os.path.join(networks_dir, filename)
            if os.path.isfile(path):
                match = None
                match = re.match(r'%s.lua' % network, filename)
                if match:
                    with open(path) as infile:
                        return infile.read()
        # return None if not found
        return None

    @override
    def get_network_from_desc(self, network_desc):
        """
        return network object from a string representation
        """
        # return the same string
        return network_desc

    @override
    def get_network_from_previous(self, previous_network, use_same_dataset):
        """
        return new instance of network from previous network
        """
        # note: use_same_dataset is ignored here because for Torch, DIGITS
        # does not change the number of outputs of the last linear layer
        # to match the number of classes in the case of a classification
        # network. In order to write a flexible network description that
        # accounts for the number of classes, the `nClasses` external
        # parameter must be used, see documentation.

        # return the same network description
        return previous_network

    @override
    def get_network_from_path(self, path):
        """
        return network object from a file path
        """
        with open(path, 'r') as f:
            network = f.read()
        return network

    @override
    def validate_network(self, data):
        """
        validate a network
        """
        return True

    @override
    def get_network_visualization(self, desc):
        """
        return visualization of network
        """
        # save network description to temporary file
        temp_network_handle, temp_network_path = tempfile.mkstemp(suffix='.lua')
        os.write(temp_network_handle, desc)
        os.close(temp_network_handle)

        try:  # do this in a try..finally clause to make sure we delete the temp file
            # build command line
            torch_bin = config_value('torch')['executable']

            args = [torch_bin,
                    os.path.join(os.path.dirname(digits.__file__), 'tools', 'torch', 'main.lua'),
                    '--network=%s' % os.path.splitext(os.path.basename(temp_network_path))[0],
                    '--networkDirectory=%s' % os.path.dirname(temp_network_path),
                    '--subtractMean=none',  # we are not providing a mean image
                    '--visualizeModel=yes',
                    '--type=float'
                    ]

            # execute command
            p = subprocess.Popen(args,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 close_fds=True,
                                 )

            # TODO: need to include regular expression for MAC color codes
            regex = re.compile('\x1b\[[0-9;]*m', re.UNICODE)

            # the network description will be accumulated from the command output
            # when collecting_net_definition==True
            collecting_net_definition = False
            desc = []
            unrecognized_output = []
            while p.poll() is None:
                for line in utils.nonblocking_readlines(p.stdout):
                    if line is not None:
                        # Remove whitespace and color codes.
                        # Color codes are appended to beginning and end of line by torch binary
                        # i.e., 'th'. Check the below link for more information
                        # https://groups.google.com/forum/#!searchin/torch7/color$20codes/torch7/8O_0lSgSzuA/Ih6wYg9fgcwJ  # noqa
                        line = regex.sub('', line)
                        timestamp, level, message = TorchTrainTask.preprocess_output_torch(line.strip())
                        if message:
                            if message.startswith('Network definition'):
                                collecting_net_definition = not collecting_net_definition
                        else:
                            if collecting_net_definition:
                                desc.append(line)
                            elif len(line):
                                unrecognized_output.append(line)
                    else:
                        time.sleep(0.05)

            if not len(desc):
                # we did not find a network description
                raise NetworkVisualizationError(''.join(unrecognized_output))
            else:
                output = flask.Markup('<pre>')
                for line in desc:
                    output += flask.Markup.escape(line)
                output += flask.Markup('</pre>')
                return output
        finally:
            os.remove(temp_network_path)
