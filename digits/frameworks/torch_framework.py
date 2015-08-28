# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import digits
import re

from framework import Framework
from digits.model import tasks
from digits.model.tasks import TorchTrainTask
from digits.utils import subclass, override

@subclass
class TorchFramework(Framework):

    """
    Defines required methods to interact with the Torch framework
    """

    # short descriptive name
    NAME = 'Torch (experimental)'

    # identifier of framework class
    CLASS = 'torch'

    # whether this framework can shuffle data during training
    CAN_SHUFFLE_DATA = True

    def __init__(self):
        super(TorchFramework, self).__init__()
        # id must be unique
        self.id = self.CLASS

    @override
    def create_train_task(self, **kwargs):
        """
        create train task
        """
        return TorchTrainTask(framework_id = self.id, **kwargs)

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
    def get_network_from_previous(self, previous_network):
        """
        return new instance of network from previous network
        """
        # return the same string
        return previous_network

    @override
    def validate_network(self, data):
        """
        validate a network
        """
        return True





