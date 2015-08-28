# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re

import digits
from exceptions import BadNetworkException
from framework import Framework
from digits.model.tasks import CaffeTrainTask
from digits.utils import subclass, override

from google.protobuf import text_format
try:
    import caffe_pb2
except ImportError:
    # See issue #32
    from caffe.proto import caffe_pb2
import caffe.draw

@subclass
class CaffeFramework(Framework):

    """
    Defines required methods to interact with a framework
    """

    # short descriptive name
    NAME = 'Caffe'

    # identifier of framework class
    CLASS = 'caffe'

    # whether this framework can shuffle data during training
    CAN_SHUFFLE_DATA = False

    @override
    def __init__(self):
        super(CaffeFramework, self).__init__()
        self.id = self.CLASS

    @override
    def create_train_task(self, **kwargs):
        """
        create train task
        """
        return CaffeTrainTask(framework_id=self.id, **kwargs)

    @override
    def validate_network(self, data):
        """
        validate a network
        """
        pb = caffe_pb2.NetParameter()
        try:
            text_format.Merge(data, pb)
        except text_format.ParseError as e:
            raise BadNetworkException('Not a valid NetParameter: %s' % e)

    @override
    def get_standard_network_desc(self, network):
        """
        return description of standard network
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
    def get_network_from_previous(self, previous_network):
        """
        return new instance of network from previous network
        """
        network = caffe_pb2.NetParameter()
        network.CopyFrom(previous_network)
        # Rename the final layer
        # XXX making some assumptions about network architecture here
        ip_layers = [l for l in network.layer if l.type == 'InnerProduct']
        if len(ip_layers) > 0:
            ip_layers[-1].name = '%s_retrain' % ip_layers[-1].name
        return network

    def get_network_visualization(self, desc):
        """
        return visualization of network
        """
        net = caffe_pb2.NetParameter()
        text_format.Merge(desc, net)
        # Throws an error if name is None
        if not net.name:
            net.name = 'Network'
        return '<image src="data:image/png;base64,' + caffe.draw.draw_net(net, 'UD').encode('base64') + '" style="max-width:100%" />'

