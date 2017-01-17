# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import re

import caffe.draw
import caffe_pb2
from google.protobuf import text_format

from .errors import BadNetworkError
from .framework import Framework
import digits
from digits.config import config_value
from digits.model.tasks import CaffeTrainTask
from digits.utils import subclass, override, parse_version


@subclass
class CaffeFramework(Framework):

    """
    Defines required methods to interact with the Caffe framework
    This class can be instantiated as many times as there are compatible
    instances of Caffe
    """

    # short descriptive name
    NAME = 'Caffe'

    # identifier of framework class (intended to be the same across
    # all instances of this class)
    CLASS = 'caffe'

    # whether this framework can shuffle data during training
    CAN_SHUFFLE_DATA = False

    if config_value('caffe')['flavor'] == 'NVIDIA':
        if parse_version(config_value('caffe')['version']) > parse_version('0.14.0-alpha'):
            SUPPORTED_SOLVER_TYPES = ['SGD', 'NESTEROV', 'ADAGRAD',
                                      'RMSPROP', 'ADADELTA', 'ADAM']
        else:
            SUPPORTED_SOLVER_TYPES = ['SGD', 'NESTEROV', 'ADAGRAD']
    elif config_value('caffe')['flavor'] == 'BVLC':
        SUPPORTED_SOLVER_TYPES = ['SGD', 'NESTEROV', 'ADAGRAD',
                                  'RMSPROP', 'ADADELTA', 'ADAM']
    else:
        raise ValueError('Unknown flavor.  Support NVIDIA and BVLC flavors only.')

    SUPPORTED_DATA_TRANSFORMATION_TYPES = ['MEAN_SUBTRACTION', 'CROPPING']
    SUPPORTED_DATA_AUGMENTATION_TYPES = []

    @override
    def __init__(self):
        super(CaffeFramework, self).__init__()
        self.framework_id = self.CLASS

    @override
    def create_train_task(self, **kwargs):
        """
        create train task
        """
        return CaffeTrainTask(framework_id=self.framework_id, **kwargs)

    @override
    def validate_network(self, data):
        """
        validate a network (input data are expected to be a text
        description of the network)
        """
        pb = caffe_pb2.NetParameter()
        try:
            text_format.Merge(data, pb)
        except text_format.ParseError as e:
            raise BadNetworkError('Not a valid NetParameter: %s' % e)

    @override
    def get_standard_network_desc(self, network):
        """
        return description of standard network
        network is expected to be a instance of caffe_pb2.NetParameter
        """
        networks_dir = os.path.join(os.path.dirname(digits.__file__), 'standard-networks', self.CLASS)

        for filename in os.listdir(networks_dir):
            path = os.path.join(networks_dir, filename)
            if os.path.isfile(path):
                match = None
                match = re.match(r'%s.prototxt' % network, filename)
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
        network = caffe_pb2.NetParameter()
        text_format.Merge(network_desc, network)
        return network

    @override
    def get_network_from_previous(self, previous_network, use_same_dataset):
        """
        return new instance of network from previous network
        """
        network = caffe_pb2.NetParameter()
        network.CopyFrom(previous_network)

        if not use_same_dataset:
            # Rename the final layer
            # XXX making some assumptions about network architecture here
            ip_layers = [l for l in network.layer if l.type == 'InnerProduct']
            if len(ip_layers) > 0:
                ip_layers[-1].name = '%s_retrain' % ip_layers[-1].name

        return network

    @override
    def get_network_from_path(self, path):
        """
        return network object from a file path
        """
        network = caffe_pb2.NetParameter()

        with open(path) as infile:
            text_format.Merge(infile.read(), network)

        return network

    @override
    def get_network_visualization(self, desc):
        """
        return visualization of network
        """
        net = caffe_pb2.NetParameter()
        text_format.Merge(desc, net)
        # Throws an error if name is None
        if not net.name:
            net.name = 'Network'
        return ('<image src="data:image/png;base64,' +
                caffe.draw.draw_net(net, 'UD').encode('base64') +
                '" style="max-width:100%" />')

    @override
    def can_accumulate_gradients(self):
        if config_value('caffe')['flavor'] == 'BVLC':
            return True
        elif config_value('caffe')['flavor'] == 'NVIDIA':
            return (parse_version(config_value('caffe')['version'])
                    > parse_version('0.14.0-alpha'))
        else:
            raise ValueError('Unknown flavor.  Support NVIDIA and BVLC flavors only.')
