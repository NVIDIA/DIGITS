# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import re
import subprocess
import tempfile
import sys

from .errors import NetworkVisualizationError
from .framework import Framework
import digits
from digits import utils
from digits.model.tasks import TensorflowTrainTask
from digits.utils import subclass, override, constants


@subclass
class TensorflowFramework(Framework):
    """
    Defines required methods to interact with the Tensorflow framework
    """

    # short descriptive name
    NAME = 'Tensorflow'

    # identifier of framework class
    CLASS = 'tensorflow'

    # whether this framework can shuffle data during training
    CAN_SHUFFLE_DATA = True
    SUPPORTS_PYTHON_LAYERS_FILE = False
    SUPPORTS_TIMELINE_TRACING = True

    SUPPORTED_SOLVER_TYPES = ['SGD', 'ADADELTA', 'ADAGRAD', 'ADAGRADDA', 'MOMENTUM', 'ADAM', 'FTRL', 'RMSPROP']

    SUPPORTED_DATA_TRANSFORMATION_TYPES = ['MEAN_SUBTRACTION', 'CROPPING']
    SUPPORTED_DATA_AUGMENTATION_TYPES = ['FLIPPING', 'NOISE', 'CONTRAST', 'WHITENING', 'HSV_SHIFTING']

    def __init__(self):
        super(TensorflowFramework, self).__init__()
        # id must be unique
        self.framework_id = self.CLASS

    @override
    def create_train_task(self, **kwargs):
        """
        create train task
        """
        return TensorflowTrainTask(framework_id=self.framework_id, **kwargs)

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
                match = re.match(r'%s.py$' % network, filename)
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
        # note: use_same_dataset is ignored here because for Tensorflow, DIGITS
        # does not change the number of outputs of the last linear layer
        # to match the number of classes in the case of a classification
        # network. In order to write a flexible network description that
        # accounts for the number of classes, the `nClasses` external
        # parameter must be used, see documentation.

        # return the same network description
        return previous_network

    @override
    def validate_network(self, data):
        """
        validate a network
        """
        return True

    @override
    def get_network_visualization(self, **kwargs):
        """
        return visualization of network
        """
        desc = kwargs['desc']
        dataset = kwargs['dataset']
        solver_type = kwargs['solver_type'].lower() if kwargs['solver_type'] else None
        use_mean = kwargs['use_mean']
        crop_size = kwargs['crop_size']
        num_gpus = kwargs['num_gpus']
        if dataset is None:
            raise NetworkVisualizationError('Make sure a dataset is selected to visualize this network.')

        # save network description to temporary file
        temp_network_handle, temp_network_path = tempfile.mkstemp(suffix='.py')
        os.write(temp_network_handle, desc)
        os.close(temp_network_handle)

        # Generate a temporaty file to put the graph definition in
        _, temp_graphdef_path = tempfile.mkstemp(suffix='.pbtxt')
        # Another for the HTML
        _, temp_html_path = tempfile.mkstemp(suffix='.html')

        try:  # do this in a try..finally clause to make sure we delete the temp file
            # build command line
            args = [sys.executable,
                    os.path.join(os.path.dirname(digits.__file__), 'tools', 'tensorflow', 'main.py'),
                    '--network=%s' % os.path.basename(temp_network_path),
                    '--networkDirectory=%s' % os.path.dirname(temp_network_path),
                    '--visualizeModelPath=%s' % temp_graphdef_path,
                    '--optimization=%s' % solver_type,
                    ]

            if crop_size:
                args.append('--croplen=%s' % crop_size)

            if use_mean and use_mean != 'none':
                mean_file = dataset.get_mean_file()
                assert mean_file is not None, 'Failed to retrieve mean file.'
                args.append('--subtractMean=%s' % use_mean)
                args.append('--mean=%s' % dataset.path(mean_file))

            if hasattr(dataset, 'labels_file'):
                args.append('--labels_list=%s' % dataset.path(dataset.labels_file))

            train_feature_db_path = dataset.get_feature_db_path(constants.TRAIN_DB)
            train_label_db_path = dataset.get_label_db_path(constants.TRAIN_DB)
            val_feature_db_path = dataset.get_feature_db_path(constants.VAL_DB)
            val_label_db_path = dataset.get_label_db_path(constants.VAL_DB)

            args.append('--train_db=%s' % train_feature_db_path)
            if train_label_db_path:
                args.append('--train_labels=%s' % train_label_db_path)
            if val_feature_db_path:
                args.append('--validation_db=%s' % val_feature_db_path)
            if val_label_db_path:
                args.append('--validation_labels=%s' % val_label_db_path)

            env = os.environ.copy()
            # make only a selected number of GPUs visible. The ID is not important for just the vis
            env['CUDA_VISIBLE_DEVICES'] = ",".join([str(i) for i in range(0, int(num_gpus))])

            # execute command
            p = subprocess.Popen(args,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 close_fds=True,
                                 env=env)

            stdout_log = ''
            while p.poll() is None:
                for line in utils.nonblocking_readlines(p.stdout):
                    timestamp, level, message = TensorflowTrainTask.preprocess_output_tensorflow(line.strip())
                    if line is not None:
                        stdout_log += line
            if p.returncode:
                raise NetworkVisualizationError(stdout_log)
            else:  # Success!
                return repr(str(open(temp_graphdef_path).read()))
        finally:
            os.remove(temp_network_path)
            os.remove(temp_graphdef_path)
