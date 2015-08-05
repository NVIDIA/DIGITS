"""
Author : Mohit Jain
Email  : develop13mohit@gmail.com

Desc : Class that handles the loading of a pretrained model to DIGITS.
"""

import time
import os.path
from collections import OrderedDict, namedtuple

import gevent
import flask

from digits import utils, device_query
from digits.task import Task
from digits.utils import override

# NOTE: Increment this everytime the picked object changes
PICKLE_VERSION = 2

# Used to store network outputs
NetworkOutput = namedtuple('NetworkOutput', ['kind', 'data'])


class LoadModelTask(Task):
    """
    Defines required methods for child classes
    """

    def __init__(self, **kwargs):
        """
        Keyword arguments:
        pretrained_model -- filename for a model to use for fine-tuning
        crop_size -- crop each image down to a square of this size
        """
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.crop_size = kwargs.pop('crop_size', None)
        self.channels = kwargs.pop('channels', 3) # Would be reset once we obtain the actual number of channels from the .prototxt

        super(LoadModelTask, self).__init__(**kwargs)
        self.pickver_task_loadmodel = PICKLE_VERSION

        self.snapshots = []

    def __getstate__(self):
        state = super(LoadModelTask, self).__getstate__()
        if 'snapshots' in state:
            del state['snapshots']
        return state

    def __setstate__(self, state):
        state['pickver_task_loadmodel'] = PICKLE_VERSION
        super(LoadModelTask, self).__setstate__(state)  

    def detect_snapshots(self):
        """
        Populate self.snapshots with snapshots that exist on disk
        Returns True if at least one usable snapshot is found
        """
        return False

    def snapshot_list(self):
        """
        Returns an array of arrays for creating an HTML select field
        """
        return [[s[1], 'Epoch #%s' % s[1]] for s in reversed(self.snapshots)]

    def est_next_snapshot(self):
        """
        Returns the estimated time in seconds until the next snapshot is taken
        """
        return None

    def can_view_weights(self):
        """
        Returns True if this Task can visualize the weights of each layer for a given model
        """
        raise NotImplementedError()

    def view_weights(self, model_epoch=None, layers=None):
        """
        View the weights for a specific model and layer[s]
        """
        return None

    def can_infer_one(self):
        """
        Returns True if this Task can run inference on one input
        """
        raise NotImplementedError()

    def can_view_activations(self):
        """
        Returns True if this Task can visualize the activations of a model after inference
        """
        raise NotImplementedError()

    def infer_one(self, data, model_epoch=None, layers=None):
        """
        Run inference on one input
        """
        return None

    def can_infer_many(self):
        """
        Returns True if this Task can run inference on many inputs
        """
        raise NotImplementedError()

    def infer_many(self, data, model_epoch=None):
        """
        Run inference on many inputs
        """
        return None

    def get_input_dims(self):
        """
        Returns the number of channels, crop_size expected as input.
        """
        if hasattr(self, '_input_dims') and self._input_dims:
            return self._input_dims

        input_dims = []
        try:
            input_dims.append(self.network.input_dim[1])
            input_dims.append(self.network.input_dim[2])
            self._input_dims = input_dims
            return self._input_dims
        except:
            raise NotImplementedError('Network deploy file missing input_dimension parameters')
         

    def get_labels(self):
        """
        Return a list of size = #categories and value of each element equal to its index representing a category.
        """
        # The labels might be set already
        if hasattr(self, '_labels') and self._labels and len(self._labels) > 0:
            return self._labels

        labels = []
        layers = []
        for layer in self.network.layer:
            assert layer.type not in ['MemoryData', 'HDF5Data', 'ImageData'], 'unsupported data layer type'
            if layer.type!='Data' and layer.type!='SoftmaxWithLoss' and layer.type!='Accuracy' and layer.type!='Softmax':
                layers.append(layer)

        lastLayer = layers[-1]
        try:
            nCategories = int(lastLayer.inner_product_param.num_output)
            for i in range(nCategories):
                labels.append(i)
        except:
            raise NotImplementedError('Network does not have the last fc layer. Failed to fetch #categories') 

        self._labels = labels
        return self._labels

