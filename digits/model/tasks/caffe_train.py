# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from collections import OrderedDict
import copy
import math
import operator
import os
import re
import sys
import time

from google.protobuf import text_format
import numpy as np
import platform
import scipy

from .train import TrainTask
import digits
from digits import utils
from digits.config import config_value
from digits.status import Status
from digits.utils import subclass, override, constants
from digits.utils.filesystem import tail

# Must import after importing digit.config
import caffe
import caffe_pb2

# NOTE: Increment this every time the pickled object changes
PICKLE_VERSION = 5

# Constants
CAFFE_SOLVER_FILE = 'solver.prototxt'
CAFFE_ORIGINAL_FILE = 'original.prototxt'
CAFFE_TRAIN_VAL_FILE = 'train_val.prototxt'
CAFFE_SNAPSHOT_PREFIX = 'snapshot'
CAFFE_DEPLOY_FILE = 'deploy.prototxt'
CAFFE_PYTHON_LAYER_FILE = 'digits_python_layers.py'


@subclass
class DigitsTransformer(caffe.io.Transformer):
    """
    A subclass of caffe.io.Transformer (an old-style class)
    Handles cases when we don't want to resize inputs
    """

    def __init__(self, resize, **kwargs):
        """
        Arguments:
        resize -- whether to resize inputs to the network default
        """
        self.resize = resize
        caffe.io.Transformer.__init__(self, **kwargs)

    def preprocess(self, in_, data):
        """
        Preprocess an image
        See parent class for details
        """
        if not self.resize:
            # update target input dimension such that no resize occurs
            self.inputs[in_] = self.inputs[in_][:2] + data.shape[:2]
            # do we have a mean?
            if in_ in self.mean:
                # resize mean if necessary
                if self.mean[in_].size > 1:
                    # we are doing mean image subtraction
                    if self.mean[in_].size != data.size:
                        # mean image size is different from data size
                        # => we need to resize the mean image
                        transpose = self.transpose.get(in_)
                        if transpose is not None:
                            # detranspose
                            self.mean[in_] = self.mean[in_].transpose(
                                np.argsort(transpose))
                        self.mean[in_] = caffe.io.resize_image(
                            self.mean[in_],
                            data.shape[:2])
                        if transpose is not None:
                            # retranspose
                            self.mean[in_] = self.mean[in_].transpose(transpose)
        return caffe.io.Transformer.preprocess(self, in_, data)


@subclass
class Error(Exception):
    pass


@subclass
class CaffeTrainSanityCheckError(Error):
    """A sanity check failed"""
    pass


@subclass
class CaffeTrainTask(TrainTask):
    """
    Trains a caffe model
    """

    CAFFE_LOG = 'caffe_output.log'

    @staticmethod
    def upgrade_network(network):
        # TODO
        pass

    @staticmethod
    def set_mode(gpu):
        if gpu is not None:
            caffe.set_device(gpu)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

    def __init__(self, **kwargs):
        """
        Arguments:
        network -- a caffe NetParameter defining the network
        """
        super(CaffeTrainTask, self).__init__(**kwargs)
        self.pickver_task_caffe_train = PICKLE_VERSION

        self.current_iteration = 0

        self.loaded_snapshot_file = None
        self.loaded_snapshot_epoch = None
        self.image_mean = None
        self.solver = None

        self.solver_file = CAFFE_SOLVER_FILE
        self.model_file = CAFFE_ORIGINAL_FILE
        self.train_val_file = CAFFE_TRAIN_VAL_FILE
        self.snapshot_prefix = CAFFE_SNAPSHOT_PREFIX
        self.deploy_file = CAFFE_DEPLOY_FILE
        self.log_file = self.CAFFE_LOG

        self.digits_version = digits.__version__
        self.caffe_version = config_value('caffe')['version']
        self.caffe_flavor = config_value('caffe')['flavor']

    def __getstate__(self):
        state = super(CaffeTrainTask, self).__getstate__()

        # Don't pickle these things
        if 'caffe_log' in state:
            del state['caffe_log']
        if '_transformer' in state:
            del state['_transformer']
        if '_caffe_net' in state:
            del state['_caffe_net']

        return state

    def __setstate__(self, state):
        super(CaffeTrainTask, self).__setstate__(state)

        # Upgrade pickle file
        if state['pickver_task_caffe_train'] <= 1:
            self.caffe_log_file = self.CAFFE_LOG
        if state['pickver_task_caffe_train'] <= 2:
            if hasattr(self, 'caffe_log_file'):
                self.log_file = self.caffe_log_file
            else:
                self.log_file = None
            self.framework_id = 'caffe'
        if state['pickver_task_caffe_train'] <= 3:
            try:
                import caffe.proto.caffe_pb2
                if isinstance(self.network, caffe.proto.caffe_pb2.NetParameter):
                    # Convert from NetParameter to string back to NetParameter
                    #   to avoid this error:
                    # TypeError: Parameter to MergeFrom() must be instance of
                    #   same class: expected caffe_pb2.NetParameter got
                    #   caffe.proto.caffe_pb2.NetParameter.
                    fixed = caffe_pb2.NetParameter()
                    text_format.Merge(
                        text_format.MessageToString(self.network),
                        fixed,
                    )
                    self.network = fixed
            except ImportError:
                # If caffe.proto.caffe_pb2 can't be imported, then you're
                # probably on a platform where that was never possible.
                # So you can't need this upgrade and we can ignore the error.
                pass

        if state['pickver_task_caffe_train'] <= 4:
            if hasattr(self, "original_file"):
                self.model_file = self.original_file
                del self.original_file
            else:
                self.model_file = None

        self.pickver_task_caffe_train = PICKLE_VERSION

        # Make changes to self
        self.loaded_snapshot_file = None
        self.loaded_snapshot_epoch = None

        # These things don't get pickled
        self.image_mean = None

    # Task overrides

    @override
    def name(self):
        return 'Train Caffe Model'

    @override
    def before_run(self):
        super(CaffeTrainTask, self).before_run()

        if isinstance(self.job, digits.model.images.classification.ImageClassificationModelJob):
            self.save_files_classification()
        elif isinstance(self.job, digits.model.images.generic.GenericImageModelJob):
            self.save_files_generic()
        else:
            raise NotImplementedError

        self.caffe_log = open(self.path(self.CAFFE_LOG), 'a')
        self.saving_snapshot = False
        self.receiving_train_output = False
        self.receiving_val_output = False
        self.last_train_update = None
        return True

    def get_mean_image(self, mean_file, resize=False):
        mean_image = None
        with open(self.dataset.path(mean_file), 'rb') as f:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(f.read())

            mean_image = np.reshape(blob.data,
                                    (
                                        self.dataset.get_feature_dims()[2],
                                        self.dataset.get_feature_dims()[0],
                                        self.dataset.get_feature_dims()[1],
                                    )
                                    )

            # Resize the mean image if crop_size exists
            if mean_image is not None and resize:
                # Get the image size needed
                network = caffe_pb2.NetParameter()
                with open(self.path(self.deploy_file)) as infile:
                    text_format.Merge(infile.read(), network)

                if network.input_shape:
                    data_shape = network.input_shape[0].dim
                else:
                    data_shape = network.input_dim[:4]
                assert len(data_shape) == 4, 'Bad data shape.'

                # Get the image
                mean_image = mean_image.astype('uint8')
                mean_image = mean_image.transpose(1, 2, 0)

                shape = list(mean_image.shape)
                # imresize will not resize if the depth is anything
                # other than 3 or 4.  If it's 1, imresize expects an
                # array.
                if (len(shape) == 2 or (len(shape) == 3 and (shape[2] == 3 or shape[2] == 4))):
                    mean_image = scipy.misc.imresize(mean_image, (data_shape[2], data_shape[3]))
                else:
                    mean_image = scipy.misc.imresize(mean_image[:, :, 0],
                                                     (data_shape[2], data_shape[3]))
                    mean_image = np.expand_dims(mean_image, axis=2)
                mean_image = mean_image.transpose(2, 0, 1)
                mean_image = mean_image.astype('float')

        return mean_image

    def get_mean_pixel(self, mean_file):
        mean_image = self.get_mean_image(mean_file)
        mean_pixel = None
        if mean_image is not None:
            mean_pixel = mean_image.mean(1).mean(1)
        return mean_pixel

    def set_mean_value(self, layer, mean_pixel):
        # remove any values that may already be in the network
        if layer.transform_param.HasField('mean_file'):
            layer.transform_param.ClearField('mean_file')
            self.logger.warning('Ignoring mean_file from network ...')

        if len(layer.transform_param.mean_value) > 0:
            layer.transform_param.ClearField('mean_value')
            self.logger.warning('Ignoring mean_value from network ...')

        layer.transform_param.mean_value.extend(list(mean_pixel))

    def set_mean_file(self, layer, mean_file):
        # remove any values that may already be in the network
        if layer.transform_param.HasField('mean_file'):
            layer.transform_param.ClearField('mean_file')
            self.logger.warning('Ignoring mean_file from network ...')

        if len(layer.transform_param.mean_value) > 0:
            layer.transform_param.ClearField('mean_value')
            self.logger.warning('Ignoring mean_value from network ...')

        layer.transform_param.mean_file = mean_file

    # TODO merge these monolithic save_files functions
    def save_files_classification(self):
        """
        Save solver, train_val and deploy files to disk
        """
        # Save the origin network to file:
        with open(self.path(self.model_file), 'w') as outfile:
            text_format.PrintMessage(self.network, outfile)

        network = cleanedUpClassificationNetwork(self.network, len(self.get_labels()))
        data_layers, train_val_layers, deploy_layers = filterLayersByState(network)

        # Write train_val file

        train_val_network = caffe_pb2.NetParameter()

        # Data layers
        # TODO clean this up

        train_data_layer = None
        val_data_layer = None

        for layer in data_layers.layer:
            for rule in layer.include:
                if rule.phase == caffe_pb2.TRAIN:
                    assert train_data_layer is None, 'cannot specify two train data layers'
                    train_data_layer = layer
                elif rule.phase == caffe_pb2.TEST:
                    assert val_data_layer is None, 'cannot specify two test data layers'
                    val_data_layer = layer

        if train_data_layer is None:
            assert val_data_layer is None, 'cannot specify a test data layer without a train data layer'

        dataset_backend = self.dataset.get_backend()
        has_val_set = self.dataset.get_entry_count(constants.VAL_DB) > 0

        if train_data_layer is not None:
            if dataset_backend == 'lmdb':
                assert train_data_layer.type == 'Data', 'expecting a Data layer'
            elif dataset_backend == 'hdf5':
                assert train_data_layer.type == 'HDF5Data', 'expecting an HDF5Data layer'
            if dataset_backend == 'lmdb' and train_data_layer.HasField('data_param'):
                assert not train_data_layer.data_param.HasField('source'), "don't set the data_param.source"
                assert not train_data_layer.data_param.HasField('backend'), "don't set the data_param.backend"
            if dataset_backend == 'hdf5' and train_data_layer.HasField('hdf5_data_param'):
                assert not train_data_layer.hdf5_data_param.HasField('source'), "don't set the hdf5_data_param.source"
            max_crop_size = min(self.dataset.get_feature_dims()[0], self.dataset.get_feature_dims()[1])
            if self.crop_size:
                assert dataset_backend != 'hdf5', 'HDF5Data layer does not support cropping'
                assert self.crop_size <= max_crop_size, 'crop_size is larger than the image size'
                train_data_layer.transform_param.crop_size = self.crop_size
            elif train_data_layer.transform_param.HasField('crop_size'):
                cs = train_data_layer.transform_param.crop_size
                if cs > max_crop_size:
                    # don't throw an error here
                    cs = max_crop_size
                train_data_layer.transform_param.crop_size = cs
                self.crop_size = cs
            train_val_network.layer.add().CopyFrom(train_data_layer)
            train_data_layer = train_val_network.layer[-1]
            if val_data_layer is not None and has_val_set:
                if dataset_backend == 'lmdb':
                    assert val_data_layer.type == 'Data', 'expecting a Data layer'
                elif dataset_backend == 'hdf5':
                    assert val_data_layer.type == 'HDF5Data', 'expecting an HDF5Data layer'
                if dataset_backend == 'lmdb' and val_data_layer.HasField('data_param'):
                    assert not val_data_layer.data_param.HasField('source'), "don't set the data_param.source"
                    assert not val_data_layer.data_param.HasField('backend'), "don't set the data_param.backend"
                if dataset_backend == 'hdf5' and val_data_layer.HasField('hdf5_data_param'):
                    assert not val_data_layer.hdf5_data_param.HasField('source'), "don't set the hdf5_data_param.source"
                if self.crop_size:
                    # use our error checking from the train layer
                    val_data_layer.transform_param.crop_size = self.crop_size
                train_val_network.layer.add().CopyFrom(val_data_layer)
                val_data_layer = train_val_network.layer[-1]
        else:
            layer_type = 'Data'
            if dataset_backend == 'hdf5':
                layer_type = 'HDF5Data'
            train_data_layer = train_val_network.layer.add(type=layer_type, name='data')
            train_data_layer.top.append('data')
            train_data_layer.top.append('label')
            train_data_layer.include.add(phase=caffe_pb2.TRAIN)
            if dataset_backend == 'lmdb':
                train_data_layer.data_param.batch_size = constants.DEFAULT_BATCH_SIZE
            elif dataset_backend == 'hdf5':
                train_data_layer.hdf5_data_param.batch_size = constants.DEFAULT_BATCH_SIZE
            if self.crop_size:
                assert dataset_backend != 'hdf5', 'HDF5Data layer does not support cropping'
                train_data_layer.transform_param.crop_size = self.crop_size
            if has_val_set:
                val_data_layer = train_val_network.layer.add(type=layer_type, name='data')
                val_data_layer.top.append('data')
                val_data_layer.top.append('label')
                val_data_layer.include.add(phase=caffe_pb2.TEST)
                if dataset_backend == 'lmdb':
                    val_data_layer.data_param.batch_size = constants.DEFAULT_BATCH_SIZE
                elif dataset_backend == 'hdf5':
                    val_data_layer.hdf5_data_param.batch_size = constants.DEFAULT_BATCH_SIZE
                if self.crop_size:
                    val_data_layer.transform_param.crop_size = self.crop_size
        if dataset_backend == 'lmdb':
            train_data_layer.data_param.source = self.dataset.get_feature_db_path(constants.TRAIN_DB)
            train_data_layer.data_param.backend = caffe_pb2.DataParameter.LMDB
            if val_data_layer is not None and has_val_set:
                val_data_layer.data_param.source = self.dataset.get_feature_db_path(constants.VAL_DB)
                val_data_layer.data_param.backend = caffe_pb2.DataParameter.LMDB
        elif dataset_backend == 'hdf5':
            train_data_layer.hdf5_data_param.source = os.path.join(
                self.dataset.get_feature_db_path(constants.TRAIN_DB), 'list.txt')
            if val_data_layer is not None and has_val_set:
                val_data_layer.hdf5_data_param.source = os.path.join(
                    self.dataset.get_feature_db_path(constants.VAL_DB), 'list.txt')

        if self.use_mean == 'pixel':
            assert dataset_backend != 'hdf5', 'HDF5Data layer does not support mean subtraction'
            mean_pixel = self.get_mean_pixel(self.dataset.path(self.dataset.get_mean_file()))
            self.set_mean_value(train_data_layer, mean_pixel)
            if val_data_layer is not None and has_val_set:
                self.set_mean_value(val_data_layer, mean_pixel)

        elif self.use_mean == 'image':
            self.set_mean_file(train_data_layer, self.dataset.path(self.dataset.get_mean_file()))
            if val_data_layer is not None and has_val_set:
                self.set_mean_file(val_data_layer, self.dataset.path(self.dataset.get_mean_file()))

        if self.batch_size:
            if dataset_backend == 'lmdb':
                train_data_layer.data_param.batch_size = self.batch_size
                if val_data_layer is not None and has_val_set:
                    val_data_layer.data_param.batch_size = self.batch_size
            elif dataset_backend == 'hdf5':
                train_data_layer.hdf5_data_param.batch_size = self.batch_size
                if val_data_layer is not None and has_val_set:
                    val_data_layer.hdf5_data_param.batch_size = self.batch_size
        else:
            if dataset_backend == 'lmdb':
                if not train_data_layer.data_param.HasField('batch_size'):
                    train_data_layer.data_param.batch_size = constants.DEFAULT_BATCH_SIZE
                if val_data_layer is not None and has_val_set and not val_data_layer.data_param.HasField('batch_size'):
                    val_data_layer.data_param.batch_size = constants.DEFAULT_BATCH_SIZE
            elif dataset_backend == 'hdf5':
                if not train_data_layer.hdf5_data_param.HasField('batch_size'):
                    train_data_layer.hdf5_data_param.batch_size = constants.DEFAULT_BATCH_SIZE
                if (val_data_layer is not None and has_val_set and
                        not val_data_layer.hdf5_data_param.HasField('batch_size')):
                    val_data_layer.hdf5_data_param.batch_size = constants.DEFAULT_BATCH_SIZE

        # Non-data layers
        train_val_network.MergeFrom(train_val_layers)

        # Write to file
        with open(self.path(self.train_val_file), 'w') as outfile:
            text_format.PrintMessage(train_val_network, outfile)

        # network sanity checks
        self.logger.debug("Network sanity check - train")
        CaffeTrainTask.net_sanity_check(train_val_network, caffe_pb2.TRAIN)
        if has_val_set:
            self.logger.debug("Network sanity check - val")
            CaffeTrainTask.net_sanity_check(train_val_network, caffe_pb2.TEST)

        # Write deploy file

        deploy_network = caffe_pb2.NetParameter()

        # Input
        deploy_network.input.append('data')
        shape = deploy_network.input_shape.add()
        shape.dim.append(1)
        shape.dim.append(self.dataset.get_feature_dims()[2])
        if self.crop_size:
            shape.dim.append(self.crop_size)
            shape.dim.append(self.crop_size)
        else:
            shape.dim.append(self.dataset.get_feature_dims()[0])
            shape.dim.append(self.dataset.get_feature_dims()[1])

        # Layers
        deploy_network.MergeFrom(deploy_layers)

        # Write to file
        with open(self.path(self.deploy_file), 'w') as outfile:
            text_format.PrintMessage(deploy_network, outfile)

        # network sanity checks
        self.logger.debug("Network sanity check - deploy")
        CaffeTrainTask.net_sanity_check(deploy_network, caffe_pb2.TEST)
        found_softmax = False
        for layer in deploy_network.layer:
            if layer.type == 'Softmax':
                found_softmax = True
                break
        assert found_softmax, \
            ('Your deploy network is missing a Softmax layer! '
             'Read the documentation for custom networks and/or look at the standard networks for examples.')

        # Write solver file

        solver = caffe_pb2.SolverParameter()
        # get enum value for solver type
        solver.solver_type = getattr(solver, self.solver_type)
        solver.net = self.train_val_file

        # Set CPU/GPU mode
        if config_value('caffe')['cuda_enabled'] and \
                bool(config_value('gpu_list')):
            solver.solver_mode = caffe_pb2.SolverParameter.GPU
        else:
            solver.solver_mode = caffe_pb2.SolverParameter.CPU

        solver.snapshot_prefix = self.snapshot_prefix

        # Batch accumulation
        from digits.frameworks import CaffeFramework
        if self.batch_accumulation and CaffeFramework().can_accumulate_gradients():
            solver.iter_size = self.batch_accumulation

        # Epochs -> Iterations
        train_iter = int(math.ceil(
            float(self.dataset.get_entry_count(constants.TRAIN_DB)) /
            (train_data_layer.data_param.batch_size * solver.iter_size)
        ))
        solver.max_iter = train_iter * self.train_epochs
        snapshot_interval = self.snapshot_interval * train_iter
        if 0 < snapshot_interval <= 1:
            solver.snapshot = 1  # don't round down
        elif 1 < snapshot_interval < solver.max_iter:
            solver.snapshot = int(snapshot_interval)
        else:
            solver.snapshot = 0  # only take one snapshot at the end

        if has_val_set and self.val_interval:
            solver.test_iter.append(
                int(math.ceil(float(self.dataset.get_entry_count(constants.VAL_DB)) /
                              val_data_layer.data_param.batch_size)))
            val_interval = self.val_interval * train_iter
            if 0 < val_interval <= 1:
                solver.test_interval = 1  # don't round down
            elif 1 < val_interval < solver.max_iter:
                solver.test_interval = int(val_interval)
            else:
                solver.test_interval = solver.max_iter  # only test once at the end

        # Learning rate
        solver.base_lr = self.learning_rate
        solver.lr_policy = self.lr_policy['policy']
        scale = float(solver.max_iter) / 100.0
        if solver.lr_policy == 'fixed':
            pass
        elif solver.lr_policy == 'step':
            # stepsize = stepsize * scale
            solver.stepsize = int(math.ceil(float(self.lr_policy['stepsize']) * scale))
            solver.gamma = self.lr_policy['gamma']
        elif solver.lr_policy == 'multistep':
            for value in self.lr_policy['stepvalue'].split(','):
                # stepvalue = stepvalue * scale
                solver.stepvalue.append(int(math.ceil(float(value) * scale)))
            solver.gamma = self.lr_policy['gamma']
        elif solver.lr_policy == 'exp':
            # gamma = gamma^(1/scale)
            solver.gamma = math.pow(self.lr_policy['gamma'], 1.0 / scale)
        elif solver.lr_policy == 'inv':
            # gamma = gamma / scale
            solver.gamma = self.lr_policy['gamma'] / scale
            solver.power = self.lr_policy['power']
        elif solver.lr_policy == 'poly':
            solver.power = self.lr_policy['power']
        elif solver.lr_policy == 'sigmoid':
            # gamma = -gamma / scale
            solver.gamma = -1.0 * self.lr_policy['gamma'] / scale
            # stepsize = stepsize * scale
            solver.stepsize = int(math.ceil(float(self.lr_policy['stepsize']) * scale))
        else:
            raise Exception('Unknown lr_policy: "%s"' % solver.lr_policy)

        # These solver types don't support momentum
        unsupported = [solver.ADAGRAD]
        try:
            unsupported.append(solver.RMSPROP)
        except AttributeError:
            pass

        if solver.solver_type not in unsupported:
            solver.momentum = 0.9
        solver.weight_decay = solver.base_lr / 100.0

        # solver specific values
        if solver.solver_type == solver.RMSPROP:
            solver.rms_decay = self.rms_decay

        # Display 8x per epoch, or once per 5000 images, whichever is more frequent
        solver.display = max(1, min(
            int(math.floor(float(solver.max_iter) / (self.train_epochs * 8))),
            int(math.ceil(5000.0 / (train_data_layer.data_param.batch_size * solver.iter_size)))
        ))

        if self.random_seed is not None:
            solver.random_seed = self.random_seed

        with open(self.path(self.solver_file), 'w') as outfile:
            text_format.PrintMessage(solver, outfile)
        self.solver = solver  # save for later

        return True

    def save_files_generic(self):
        """
        Save solver, train_val and deploy files to disk
        """
        train_feature_db_path = self.dataset.get_feature_db_path(constants.TRAIN_DB)
        train_label_db_path = self.dataset.get_label_db_path(constants.TRAIN_DB)
        val_feature_db_path = self.dataset.get_feature_db_path(constants.VAL_DB)
        val_label_db_path = self.dataset.get_label_db_path(constants.VAL_DB)

        assert train_feature_db_path is not None, 'Training images are required'

        # Save the origin network to file:
        with open(self.path(self.model_file), 'w') as outfile:
            text_format.PrintMessage(self.network, outfile)

        # Split up train_val and deploy layers

        network = cleanedUpGenericNetwork(self.network)
        data_layers, train_val_layers, deploy_layers = filterLayersByState(network)

        # Write train_val file

        train_val_network = caffe_pb2.NetParameter()

        # Data layers
        # TODO clean this up

        train_image_data_layer = None
        train_label_data_layer = None
        val_image_data_layer = None
        val_label_data_layer = None

        # Find the existing Data layers
        for layer in data_layers.layer:
            for rule in layer.include:
                if rule.phase == caffe_pb2.TRAIN:
                    for top_name in layer.top:
                        if 'data' in top_name:
                            assert train_image_data_layer is None, \
                                'cannot specify two train image data layers'
                            train_image_data_layer = layer
                        elif 'label' in top_name:
                            assert train_label_data_layer is None, \
                                'cannot specify two train label data layers'
                            train_label_data_layer = layer
                elif rule.phase == caffe_pb2.TEST:
                    for top_name in layer.top:
                        if 'data' in top_name:
                            assert val_image_data_layer is None, \
                                'cannot specify two val image data layers'
                            val_image_data_layer = layer
                        elif 'label' in top_name:
                            assert val_label_data_layer is None, \
                                'cannot specify two val label data layers'
                            val_label_data_layer = layer

        # Create and add the Data layers
        # (uses info from existing data layers, where possible)
        train_image_data_layer = self.make_generic_data_layer(
            train_feature_db_path, train_image_data_layer, 'data', 'data', caffe_pb2.TRAIN)
        if train_image_data_layer is not None:
            train_val_network.layer.add().CopyFrom(train_image_data_layer)

        train_label_data_layer = self.make_generic_data_layer(
            train_label_db_path, train_label_data_layer, 'label', 'label', caffe_pb2.TRAIN)
        if train_label_data_layer is not None:
            train_val_network.layer.add().CopyFrom(train_label_data_layer)

        val_image_data_layer = self.make_generic_data_layer(
            val_feature_db_path, val_image_data_layer, 'data', 'data', caffe_pb2.TEST)
        if val_image_data_layer is not None:
            train_val_network.layer.add().CopyFrom(val_image_data_layer)

        val_label_data_layer = self.make_generic_data_layer(
            val_label_db_path, val_label_data_layer, 'label', 'label', caffe_pb2.TEST)
        if val_label_data_layer is not None:
            train_val_network.layer.add().CopyFrom(val_label_data_layer)

        # Add non-data layers
        train_val_network.MergeFrom(train_val_layers)

        # Write to file
        with open(self.path(self.train_val_file), 'w') as outfile:
            text_format.PrintMessage(train_val_network, outfile)

        # network sanity checks
        self.logger.debug("Network sanity check - train")
        CaffeTrainTask.net_sanity_check(train_val_network, caffe_pb2.TRAIN)
        if val_image_data_layer is not None:
            self.logger.debug("Network sanity check - val")
            CaffeTrainTask.net_sanity_check(train_val_network, caffe_pb2.TEST)

        # Write deploy file

        deploy_network = caffe_pb2.NetParameter()

        # Input
        deploy_network.input.append('data')
        shape = deploy_network.input_shape.add()
        shape.dim.append(1)
        shape.dim.append(self.dataset.get_feature_dims()[2])  # channels
        if train_image_data_layer.transform_param.HasField('crop_size'):
            shape.dim.append(
                train_image_data_layer.transform_param.crop_size)
            shape.dim.append(
                train_image_data_layer.transform_param.crop_size)
        else:
            shape.dim.append(self.dataset.get_feature_dims()[0])  # height
            shape.dim.append(self.dataset.get_feature_dims()[1])  # width

        # Layers
        deploy_network.MergeFrom(deploy_layers)

        # Write to file
        with open(self.path(self.deploy_file), 'w') as outfile:
            text_format.PrintMessage(deploy_network, outfile)

        # network sanity checks
        self.logger.debug("Network sanity check - deploy")
        CaffeTrainTask.net_sanity_check(deploy_network, caffe_pb2.TEST)

        # Write solver file

        solver = caffe_pb2.SolverParameter()
        # get enum value for solver type
        solver.solver_type = getattr(solver, self.solver_type)
        solver.net = self.train_val_file

        # Set CPU/GPU mode
        if config_value('caffe')['cuda_enabled'] and \
                bool(config_value('gpu_list')):
            solver.solver_mode = caffe_pb2.SolverParameter.GPU
        else:
            solver.solver_mode = caffe_pb2.SolverParameter.CPU

        solver.snapshot_prefix = self.snapshot_prefix

        # Batch accumulation
        from digits.frameworks import CaffeFramework
        if self.batch_accumulation and CaffeFramework().can_accumulate_gradients():
            solver.iter_size = self.batch_accumulation

        # Epochs -> Iterations
        train_iter = int(math.ceil(
            float(self.dataset.get_entry_count(constants.TRAIN_DB)) /
            (train_image_data_layer.data_param.batch_size * solver.iter_size)
        ))
        solver.max_iter = train_iter * self.train_epochs
        snapshot_interval = self.snapshot_interval * train_iter
        if 0 < snapshot_interval <= 1:
            solver.snapshot = 1  # don't round down
        elif 1 < snapshot_interval < solver.max_iter:
            solver.snapshot = int(snapshot_interval)
        else:
            solver.snapshot = 0  # only take one snapshot at the end

        if val_image_data_layer:
            solver.test_iter.append(int(math.ceil(float(self.dataset.get_entry_count(
                constants.VAL_DB)) / val_image_data_layer.data_param.batch_size)))
            val_interval = self.val_interval * train_iter
            if 0 < val_interval <= 1:
                solver.test_interval = 1  # don't round down
            elif 1 < val_interval < solver.max_iter:
                solver.test_interval = int(val_interval)
            else:
                solver.test_interval = solver.max_iter  # only test once at the end

        # Learning rate
        solver.base_lr = self.learning_rate
        solver.lr_policy = self.lr_policy['policy']
        scale = float(solver.max_iter) / 100.0
        if solver.lr_policy == 'fixed':
            pass
        elif solver.lr_policy == 'step':
            # stepsize = stepsize * scale
            solver.stepsize = int(math.ceil(float(self.lr_policy['stepsize']) * scale))
            solver.gamma = self.lr_policy['gamma']
        elif solver.lr_policy == 'multistep':
            for value in self.lr_policy['stepvalue'].split(','):
                # stepvalue = stepvalue * scale
                solver.stepvalue.append(int(math.ceil(float(value) * scale)))
            solver.gamma = self.lr_policy['gamma']
        elif solver.lr_policy == 'exp':
            # gamma = gamma^(1/scale)
            solver.gamma = math.pow(self.lr_policy['gamma'], 1.0 / scale)
        elif solver.lr_policy == 'inv':
            # gamma = gamma / scale
            solver.gamma = self.lr_policy['gamma'] / scale
            solver.power = self.lr_policy['power']
        elif solver.lr_policy == 'poly':
            solver.power = self.lr_policy['power']
        elif solver.lr_policy == 'sigmoid':
            # gamma = -gamma / scale
            solver.gamma = -1.0 * self.lr_policy['gamma'] / scale
            # stepsize = stepsize * scale
            solver.stepsize = int(math.ceil(float(self.lr_policy['stepsize']) * scale))
        else:
            raise Exception('Unknown lr_policy: "%s"' % solver.lr_policy)

        # These solver types don't support momentum
        unsupported = [solver.ADAGRAD]
        try:
            unsupported.append(solver.RMSPROP)
        except AttributeError:
            pass

        if solver.solver_type not in unsupported:
            solver.momentum = 0.9
        solver.weight_decay = solver.base_lr / 100.0

        # Display 8x per epoch, or once per 5000 images, whichever is more frequent
        solver.display = max(1, min(
            int(math.floor(float(solver.max_iter) / (self.train_epochs * 8))),
            int(math.ceil(5000.0 / (train_image_data_layer.data_param.batch_size * solver.iter_size)))
        ))

        if self.random_seed is not None:
            solver.random_seed = self.random_seed

        with open(self.path(self.solver_file), 'w') as outfile:
            text_format.PrintMessage(solver, outfile)
        self.solver = solver  # save for later

        return True

    def make_generic_data_layer(self, db_path, orig_layer, name, top, phase):
        """
        Utility within save_files_generic for creating a Data layer
        Returns a LayerParameter (or None)

        Arguments:
        db_path -- path to database (or None)
        orig_layer -- a LayerParameter supplied by the user (or None)
        """
        if db_path is None:
            # TODO allow user to specify a standard data layer even if it doesn't exist in the dataset
            return None
        layer = caffe_pb2.LayerParameter()
        if orig_layer is not None:
            layer.CopyFrom(orig_layer)
        layer.type = 'Data'
        if not layer.HasField('name'):
            layer.name = name
        if not len(layer.top):
            layer.top.append(top)
        layer.ClearField('include')
        layer.include.add(phase=phase)

        # source
        if layer.data_param.HasField('source'):
            self.logger.warning('Ignoring data_param.source ...')
        layer.data_param.source = db_path
        if layer.data_param.HasField('backend'):
            self.logger.warning('Ignoring data_param.backend ...')
        layer.data_param.backend = caffe_pb2.DataParameter.LMDB

        # batch size
        if not layer.data_param.HasField('batch_size'):
            layer.data_param.batch_size = constants.DEFAULT_BATCH_SIZE
        if self.batch_size:
            layer.data_param.batch_size = self.batch_size

        # mean
        if name == 'data' and self.dataset.get_mean_file():
            if self.use_mean == 'pixel':
                mean_pixel = self.get_mean_pixel(self.dataset.path(self.dataset.get_mean_file()))
                # remove any values that may already be in the network
                self.set_mean_value(layer, mean_pixel)
            elif self.use_mean == 'image':
                self.set_mean_file(layer, self.dataset.path(self.dataset.get_mean_file()))

        # crop size
        if name == 'data' and self.crop_size:
            max_crop_size = min(self.dataset.get_feature_dims()[0], self.dataset.get_feature_dims()[1])
            assert self.crop_size <= max_crop_size, 'crop_size is larger than the image size'
            layer.transform_param.crop_size = self.crop_size
        return layer

    def iteration_to_epoch(self, it):
        return float(it * self.train_epochs) / self.solver.max_iter

    @override
    def task_arguments(self, resources, env):
        """
        Generate Caffe command line options or, in certain cases, pycaffe Python script
        Returns a list of strings

        Arguments:
        resources -- dict of available task resources
        env -- dict of environment variables
        """
        if platform.system() == 'Windows':
            if any([layer.type == 'Python' for layer in self.network.layer]):
                # Arriving here because the network includes Python Layer and we are running inside Windows.
                # We can not invoke caffe.exe and need to fallback to pycaffe
                # https://github.com/Microsoft/caffe/issues/87
                # TODO: Remove this once caffe.exe works fine with Python Layer
                win_python_layer_gpu_id = None
                if 'gpus' in resources:
                    n_gpus = len(resources['gpus'])
                    if n_gpus > 1:
                        raise Exception('Please select single GPU when running in Windows with Python layer.')
                    elif n_gpus == 1:
                        win_python_layer_gpu_id = resources['gpus'][0][0]
                # We know which GPU to use, call helper to create the script
                return self._pycaffe_args(win_python_layer_gpu_id)

        # Not in Windows, or in Windows but no Python Layer
        # This is the normal path
        args = [config_value('caffe')['executable'],
                'train',
                '--solver=%s' % self.path(self.solver_file),
                ]

        if 'gpus' in resources:
            identifiers = []
            for identifier, value in resources['gpus']:
                identifiers.append(identifier)
            if len(identifiers) == 1:
                args.append('--gpu=%s' % identifiers[0])
            elif len(identifiers) > 1:
                if config_value('caffe')['flavor'] == 'NVIDIA':
                    if (utils.parse_version(config_value('caffe')['version'])
                            < utils.parse_version('0.14.0-alpha')):
                        # Prior to version 0.14, NVcaffe used the --gpus switch
                        args.append('--gpus=%s' % ','.join(identifiers))
                    else:
                        args.append('--gpu=%s' % ','.join(identifiers))
                elif config_value('caffe')['flavor'] == 'BVLC':
                    args.append('--gpu=%s' % ','.join(identifiers))
                else:
                    raise ValueError('Unknown flavor.  Support NVIDIA and BVLC flavors only.')
        if self.pretrained_model:
            args.append('--weights=%s' % ','.join(map(lambda x: self.path(x),
                                                      self.pretrained_model.split(os.path.pathsep))))
        return args

    def _pycaffe_args(self, gpu_id):
        """
        Helper to generate pycaffe Python script
        Returns a list of strings
        Throws ValueError if self.solver_type is not recognized

        Arguments:
        gpu_id -- the GPU device id to use
        """
        # TODO: Remove this once caffe.exe works fine with Python Layer
        solver_type_mapping = {
            'ADADELTA': 'AdaDeltaSolver',
            'ADAGRAD': 'AdaGradSolver',
            'ADAM': 'AdamSolver',
            'NESTEROV': 'NesterovSolver',
            'RMSPROP': 'RMSPropSolver',
            'SGD': 'SGDSolver'}
        try:
            solver_type = solver_type_mapping[self.solver_type]
        except KeyError:
            raise ValueError("Unknown solver type {}.".format(self.solver_type))
        if gpu_id is not None:
            gpu_script = "caffe.set_device({id});caffe.set_mode_gpu();".format(id=gpu_id)
        else:
            gpu_script = "caffe.set_mode_cpu();"
        loading_script = ""
        if self.pretrained_model:
            weight_files = map(lambda x: self.path(x), self.pretrained_model.split(os.path.pathsep))
            for weight_file in weight_files:
                loading_script = loading_script + "solv.net.copy_from('{weight}');".format(weight=weight_file)
        command_script =\
            "import caffe;" \
            "{gpu_script}" \
            "solv=caffe.{solver}('{solver_file}');" \
            "{loading_script}" \
            "solv.solve()" \
            .format(gpu_script=gpu_script,
                    solver=solver_type,
                    solver_file=self.solver_file, loading_script=loading_script)
        args = [sys.executable + ' -c ' + '\"' + command_script + '\"']
        return args

    @override
    def process_output(self, line):
        float_exp = '(NaN|[-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?)'

        self.caffe_log.write('%s\n' % line)
        self.caffe_log.flush()
        # parse caffe output
        timestamp, level, message = self.preprocess_output_caffe(line)
        if not message:
            return True

        # iteration updates
        match = re.match(r'Iteration (\d+)', message)
        if match:
            i = int(match.group(1))
            self.new_iteration(i)

        # net output
        match = re.match(r'(Train|Test) net output #(\d+): (\S*) = %s' % float_exp, message, flags=re.IGNORECASE)
        if match:
            phase = match.group(1)
            # index = int(match.group(2))
            name = match.group(3)
            value = match.group(4)
            assert value.lower() != 'nan', \
                'Network outputted NaN for "%s" (%s phase). Try decreasing your learning rate.' % (name, phase)
            value = float(value)

            # Find the layer type
            kind = '?'
            for layer in self.network.layer:
                if name in layer.top:
                    kind = layer.type
                    break

            if phase.lower() == 'train':
                self.save_train_output(name, kind, value)
            elif phase.lower() == 'test':
                self.save_val_output(name, kind, value)
            return True

        # learning rate updates
        match = re.match(r'Iteration (\d+).*lr = %s' % float_exp, message, flags=re.IGNORECASE)
        if match:
            i = int(match.group(1))
            lr = float(match.group(2))
            self.save_train_output('learning_rate', 'LearningRate', lr)
            return True

        # snapshot saved
        if self.saving_snapshot:
            if not message.startswith('Snapshotting solver state'):
                self.logger.warning(
                    'caffe output format seems to have changed. '
                    'Expected "Snapshotting solver state..." after "Snapshotting to..."')
            else:
                self.logger.debug('Snapshot saved.')
            self.detect_snapshots()
            self.send_snapshot_update()
            self.saving_snapshot = False
            return True

        # snapshot starting
        match = re.match(r'Snapshotting to (.*)\s*$', message)
        if match:
            self.saving_snapshot = True
            return True

        if level in ['error', 'critical']:
            self.logger.error('%s: %s' % (self.name(), message))
            self.exception = message
            return True

        return True

    def preprocess_output_caffe(self, line):
        """
        Takes line of output and parses it according to caffe's output format
        Returns (timestamp, level, message) or (None, None, None)
        """
        # NOTE: This must change when the logging format changes
        # LMMDD HH:MM:SS.MICROS pid file:lineno] message
        match = re.match(r'(\w)(\d{4} \S{8}).*]\s+(\S.*)$', line)
        if match:
            level = match.group(1)
            # add the year because caffe omits it
            timestr = '%s%s' % (time.strftime('%Y'), match.group(2))
            message = match.group(3)
            if level == 'I':
                level = 'info'
            elif level == 'W':
                level = 'warning'
            elif level == 'E':
                level = 'error'
            elif level == 'F':  # FAIL
                level = 'critical'
            timestamp = time.mktime(time.strptime(timestr, '%Y%m%d %H:%M:%S'))
            return (timestamp, level, message)
        else:
            return (None, None, None)

    def new_iteration(self, it):
        """
        Update current_iteration
        """
        if self.current_iteration == it:
            return

        self.current_iteration = it
        self.send_progress_update(self.iteration_to_epoch(it))

    def send_snapshot_update(self):
        """
        Sends socketio message about the snapshot list
        """
        from digits.webapp import socketio

        socketio.emit('task update',
                      {
                          'task': self.html_id(),
                          'update': 'snapshots',
                          'data': self.snapshot_list(),
                      },
                      namespace='/jobs',
                      room=self.job_id,
                      )

    @override
    def after_run(self):
        super(CaffeTrainTask, self).after_run()
        self.caffe_log.close()

    @override
    def after_runtime_error(self):
        if os.path.exists(self.path(self.CAFFE_LOG)):
            output = tail(self.path(self.CAFFE_LOG), 40)
            lines = []
            for line in output.split('\n'):
                # parse caffe header
                timestamp, level, message = self.preprocess_output_caffe(line)

                if message:
                    lines.append(message)
            # return the last 20 lines
            self.traceback = '\n'.join(lines[len(lines) - 20:])
            if 'DIGITS_MODE_TEST' in os.environ:
                print output

    # TrainTask overrides

    @override
    def get_task_stats(self, epoch=-1):
        """
        return a dictionary of task statistics
        """

        loc, mean_file = os.path.split(self.dataset.get_mean_file())

        stats = {
            "image dimensions": self.dataset.get_feature_dims(),
            "mean file": mean_file,
            "snapshot file": self.get_snapshot_filename(epoch),
            "solver file": self.solver_file,
            "train_val file": self.train_val_file,
            "deploy file": self.deploy_file,
            "framework": "caffe"
        }

        # These attributes only available in more recent jobs:
        if hasattr(self, "model_file"):
            if self.model_file is not None:
                stats.update({
                    "caffe flavor": self.caffe_flavor,
                    "caffe version": self.caffe_version,
                    "model file": self.model_file,
                    "digits version": self.digits_version
                })

        if hasattr(self.dataset, "resize_mode"):
            stats.update({"image resize mode": self.dataset.resize_mode})

        if hasattr(self.dataset, "labels_file"):
            stats.update({"labels file": self.dataset.labels_file})

        # Add this if python layer file exists
        if os.path.exists(os.path.join(self.job_dir, CAFFE_PYTHON_LAYER_FILE)):
            stats.update({"python layer file": CAFFE_PYTHON_LAYER_FILE})
        elif os.path.exists(os.path.join(self.job_dir, CAFFE_PYTHON_LAYER_FILE + 'c')):
            stats.update({"python layer file": CAFFE_PYTHON_LAYER_FILE + 'c'})

        return stats

    @override
    def detect_snapshots(self):
        self.snapshots = []

        snapshot_dir = os.path.join(self.job_dir, os.path.dirname(self.snapshot_prefix))
        snapshots = []
        solverstates = []

        for filename in os.listdir(snapshot_dir):
            # find models
            match = re.match(r'%s_iter_(\d+)\.caffemodel' % os.path.basename(self.snapshot_prefix), filename)
            if match:
                iteration = int(match.group(1))
                epoch = float(iteration) / (float(self.solver.max_iter) / self.train_epochs)
                # assert epoch.is_integer(), '%s is not an integer' % epoch
                epoch = round(epoch, 3)
                # if epoch is int
                if epoch == math.ceil(epoch):
                    # print epoch,math.ceil(epoch),int(epoch)
                    epoch = int(epoch)
                snapshots.append((
                    os.path.join(snapshot_dir, filename),
                    epoch
                )
                )
            # find solverstates
            match = re.match(r'%s_iter_(\d+)\.solverstate' % os.path.basename(self.snapshot_prefix), filename)
            if match:
                solverstates.append((
                    os.path.join(snapshot_dir, filename),
                    int(match.group(1))
                )
                )

        # delete all but the most recent solverstate
        for filename, iteration in sorted(solverstates, key=lambda tup: tup[1])[:-1]:
            # print 'Removing "%s"' % filename
            os.remove(filename)

        self.snapshots = sorted(snapshots, key=lambda tup: tup[1])

        return len(self.snapshots) > 0

    @override
    def est_next_snapshot(self):
        if self.status != Status.RUN or self.current_iteration == 0:
            return None
        elapsed = time.time() - self.status_updates[-1][1]
        next_snapshot_iteration = (1 + self.current_iteration // self.snapshot_interval) * self.snapshot_interval
        return (next_snapshot_iteration - self.current_iteration) * elapsed // self.current_iteration

    @override
    def can_view_weights(self):
        return False

    @override
    def infer_one(self,
                  data,
                  snapshot_epoch=None,
                  layers=None,
                  gpu=None,
                  resize=True):
        return self.infer_one_image(data,
                                    snapshot_epoch=snapshot_epoch,
                                    layers=layers,
                                    gpu=gpu,
                                    resize=resize
                                    )

    def infer_one_image(self,
                        image,
                        snapshot_epoch=None,
                        layers=None,
                        gpu=None,
                        resize=True):
        """
        Run inference on one image for a generic model
        Returns (output, visualizations)
            output -- an OrderedDict of string -> np.ndarray
            visualizations -- a list of dicts for the specified layers
        Returns (None, None) if something goes wrong

        Arguments:
        image -- an np.ndarray

        Keyword arguments:
        snapshot_epoch -- which snapshot to use
        layers -- which layer activation[s] and weight[s] to visualize
        """
        net = self.get_net(snapshot_epoch, gpu=gpu)

        # process image
        if image.ndim == 2:
            image = image[:, :, np.newaxis]

        preprocessed = self.get_transformer(resize).preprocess(
            'data', image)

        # reshape net input (if necessary)
        test_shape = (1,) + preprocessed.shape
        if net.blobs['data'].data.shape != test_shape:
            net.blobs['data'].reshape(*test_shape)

        # run inference
        net.blobs['data'].data[...] = preprocessed
        o = net.forward()

        # order outputs in prototxt order
        output = OrderedDict()
        for blob in net.blobs.keys():
            if blob in o:
                output[blob] = o[blob]

        visualizations = self.get_layer_visualizations(net, layers)
        return (output, visualizations)

    def get_layer_visualizations(self, net, layers='all'):
        """
        Returns visualizations of various layers in the network
        """
        # add visualizations
        visualizations = []
        if layers and layers != 'none':
            if layers == 'all':
                added_activations = []
                for layer in self.network.layer:
                    for bottom in layer.bottom:
                        if bottom in net.blobs and bottom not in added_activations:
                            data = net.blobs[bottom].data[0]
                            vis = utils.image.get_layer_vis_square(data,
                                                                   allow_heatmap=bool(bottom != 'data'),
                                                                   channel_order='BGR')
                            mean, std, hist = self.get_layer_statistics(data)
                            visualizations.append(
                                {
                                    'name': str(bottom),
                                    'vis_type': 'Activation',
                                    'vis': vis,
                                    'data_stats': {
                                        'shape': data.shape,
                                        'mean': mean,
                                        'stddev': std,
                                        'histogram': hist,
                                    },
                                }
                            )
                            added_activations.append(bottom)
                    if layer.name in net.params:
                        data = net.params[layer.name][0].data
                        if layer.type not in ['InnerProduct']:
                            vis = utils.image.get_layer_vis_square(data, channel_order='BGR')
                        else:
                            vis = None
                        mean, std, hist = self.get_layer_statistics(data)
                        params = net.params[layer.name]
                        weight_count = reduce(operator.mul, params[0].data.shape, 1)
                        if len(params) > 1:
                            bias_count = reduce(operator.mul, params[1].data.shape, 1)
                        else:
                            bias_count = 0
                        parameter_count = weight_count + bias_count
                        visualizations.append(
                            {
                                'name': str(layer.name),
                                'vis_type': 'Weights',
                                'layer_type': layer.type,
                                'param_count': parameter_count,
                                'vis': vis,
                                'data_stats': {
                                    'shape': data.shape,
                                    'mean': mean,
                                    'stddev': std,
                                    'histogram': hist,
                                },
                            }
                        )
                    for top in layer.top:
                        if top in net.blobs and top not in added_activations:
                            data = net.blobs[top].data[0]
                            normalize = True
                            # don't normalize softmax layers
                            if layer.type == 'Softmax':
                                normalize = False
                            vis = utils.image.get_layer_vis_square(data,
                                                                   normalize=normalize,
                                                                   allow_heatmap=bool(top != 'data'),
                                                                   channel_order='BGR')
                            mean, std, hist = self.get_layer_statistics(data)
                            visualizations.append(
                                {
                                    'name': str(top),
                                    'vis_type': 'Activation',
                                    'vis': vis,
                                    'data_stats': {
                                        'shape': data.shape,
                                        'mean': mean,
                                        'stddev': std,
                                        'histogram': hist,
                                    },
                                }
                            )
                            added_activations.append(top)
            else:
                raise NotImplementedError

        return visualizations

    def get_layer_statistics(self, data):
        """
        Returns statistics for the given layer data:
            (mean, standard deviation, histogram)
                histogram -- [y, x, ticks]

        Arguments:
        data -- a np.ndarray
        """
        # XXX These calculations can be super slow
        mean = np.mean(data).astype(np.float32)
        std = np.std(data).astype(np.float32)
        y, x = np.histogram(data, bins=20)
        y = list(y.astype(np.float32))
        ticks = x[[0, len(x) / 2, -1]]
        x = [((x[i] + x[i + 1]) / 2.0).astype(np.float32) for i in xrange(len(x) - 1)]
        ticks = list(ticks.astype(np.float32))
        return (mean, std, [y, x, ticks])

    @override
    def infer_many(self,
                   data,
                   snapshot_epoch=None,
                   gpu=None,
                   resize=True):
        return self.infer_many_images(data,
                                      snapshot_epoch=snapshot_epoch,
                                      gpu=gpu,
                                      resize=resize)

    def infer_many_images(self,
                          images,
                          snapshot_epoch=None,
                          gpu=None,
                          resize=True):
        """
        Returns a list of OrderedDict, one for each image

        Arguments:
        images -- a list of np.arrays

        Keyword arguments:
        snapshot_epoch -- which snapshot to use
        """
        net = self.get_net(snapshot_epoch, gpu=gpu)

        caffe_images = []
        for image in images:
            if image.ndim == 2:
                caffe_images.append(image[:, :, np.newaxis])
            else:
                caffe_images.append(image)

        data_shape = tuple(self.get_transformer(resize).inputs['data'])[1:]

        if self.batch_size:
            data_shape = (self.batch_size,) + data_shape
        # TODO: grab batch_size from the TEST phase in train_val network
        else:
            data_shape = (constants.DEFAULT_BATCH_SIZE,) + data_shape

        outputs = None
        for chunk in [caffe_images[x:x + data_shape[0]] for x in xrange(0, len(caffe_images), data_shape[0])]:
            new_shape = (len(chunk),) + data_shape[1:]
            if net.blobs['data'].data.shape != new_shape:
                net.blobs['data'].reshape(*new_shape)
            for index, image in enumerate(chunk):
                net.blobs['data'].data[index] = self.get_transformer(resize).preprocess(
                    'data', image)
            o = net.forward()

            # order output in prototxt order
            output = OrderedDict()
            for blob in net.blobs.keys():
                if blob in o:
                    output[blob] = o[blob]

            if outputs is None:
                outputs = copy.deepcopy(output)
            else:
                for name, blob in output.iteritems():
                    outputs[name] = np.vstack((outputs[name], blob))
            print 'Processed %s/%s images' % (len(outputs[outputs.keys()[0]]), len(caffe_images))

        return outputs

    def has_model(self):
        """
        Returns True if there is a model that can be used
        """
        return len(self.snapshots) > 0

    def get_net(self, epoch=None, gpu=None):
        """
        Returns an instance of caffe.Net

        Keyword Arguments:
        epoch -- which snapshot to load (default is -1 to load the most recently generated snapshot)
        """
        if not self.has_model():
            return False

        file_to_load = self.get_snapshot(epoch)

        # check if already loaded
        if self.loaded_snapshot_file and self.loaded_snapshot_file == file_to_load \
                and hasattr(self, '_caffe_net') and self._caffe_net is not None:
            return self._caffe_net

        CaffeTrainTask.set_mode(gpu)

        # Add job_dir to PATH to pick up any python layers used by the model
        sys.path.append(self.job_dir)

        # Attempt to force a reload of the "digits_python_layers" module
        loaded_module = sys.modules.get('digits_python_layers', None)
        if loaded_module:
            try:
                reload(loaded_module)
            except ImportError:
                # Let Caffe throw the error if the file is missing
                pass

        # Load the model
        self._caffe_net = caffe.Net(
            self.path(self.deploy_file),
            file_to_load,
            caffe.TEST)

        # Remove job_dir from PATH
        sys.path.remove(self.job_dir)

        self.loaded_snapshot_epoch = epoch
        self.loaded_snapshot_file = file_to_load

        return self._caffe_net

    def get_transformer(self, resize=True):
        """
        Returns an instance of DigitsTransformer
        Parameters:
        - resize_shape: specify shape of network (or None for network default)
        """
        # check if already loaded
        if hasattr(self, '_transformer') and self._transformer is not None:
            return self._transformer

        data_shape = None
        channel_swap = None
        mean_pixel = None
        mean_image = None

        network = caffe_pb2.NetParameter()
        with open(self.path(self.deploy_file)) as infile:
            text_format.Merge(infile.read(), network)
        if network.input_shape:
            data_shape = network.input_shape[0].dim
        else:
            data_shape = network.input_dim[:4]

        if self.dataset.get_feature_dims()[2] == 3:
            # BGR when there are three channels
            # XXX see issue #59
            channel_swap = (2, 1, 0)

        if self.dataset.get_mean_file():
            if self.use_mean == 'pixel':
                mean_pixel = self.get_mean_pixel(self.dataset.path(self.dataset.get_mean_file()))
            elif self.use_mean == 'image':
                mean_image = self.get_mean_image(self.dataset.path(self.dataset.get_mean_file()), True)

        t = DigitsTransformer(
            inputs={'data': tuple(data_shape)},
            resize=resize
        )

        # transpose to (channels, height, width)
        t.set_transpose('data', (2, 0, 1))

        if channel_swap is not None:
            # swap color channels
            t.set_channel_swap('data', channel_swap)

        # set mean
        if self.use_mean == 'pixel' and mean_pixel is not None:
            t.set_mean('data', mean_pixel)
        elif self.use_mean == 'image' and mean_image is not None:
            t.set_mean('data', mean_image)

        # t.set_raw_scale('data', 255) # [0,255] range instead of [0,1]

        self._transformer = t
        return self._transformer

    @override
    def get_model_files(self):
        """
        return paths to model files
        """
        model_files = {
            "Solver": self.solver_file,
            "Network (train/val)": self.train_val_file,
            "Network (deploy)": self.deploy_file
        }
        if os.path.exists(os.path.join(self.job_dir, CAFFE_PYTHON_LAYER_FILE)):
            model_files.update({"Python layer": os.path.join(self.job_dir, CAFFE_PYTHON_LAYER_FILE)})
        elif os.path.exists(os.path.join(self.job_dir, CAFFE_PYTHON_LAYER_FILE + 'c')):
            model_files.update({"Python layer": os.path.join(self.job_dir, CAFFE_PYTHON_LAYER_FILE + 'c')})
        if hasattr(self, "model_file"):
            if self.model_file is not None:
                model_files.update({"Network (original)": self.model_file})
        return model_files

    @override
    def get_network_desc(self):
        """
        return text description of model
        """
        return text_format.MessageToString(self.network)

    @staticmethod
    def net_sanity_check(net, phase):
        """
        Perform various sanity checks on the network, including:
        - check that all layer bottoms are included at the specified stage
        """
        assert phase == caffe_pb2.TRAIN or phase == caffe_pb2.TEST, "Unknown phase: %s" % repr(phase)
        # work out which layers and tops are included at the specified phase
        layers = []
        tops = []
        for layer in net.layer:
            if len(layer.include) > 0:
                mask = 0  # include none by default
                for rule in layer.include:
                    mask = mask | (1 << rule.phase)
            elif len(layer.exclude) > 0:
                # include and exclude rules are mutually exclusive as per Caffe spec
                mask = (1 << caffe_pb2.TRAIN) | (1 << caffe_pb2.TEST)  # include all by default
                for rule in layer.exclude:
                    mask = mask & ~(1 << rule.phase)
            else:
                mask = (1 << caffe_pb2.TRAIN) | (1 << caffe_pb2.TEST)
            if mask & (1 << phase):
                # layer will be included at this stage
                layers.append(layer)
                tops.extend(layer.top)
        # add inputs
        tops.extend(net.input)
        # now make sure all bottoms are present at this stage
        for layer in layers:
            for bottom in layer.bottom:
                if bottom not in tops:
                    raise CaffeTrainSanityCheckError(
                        "Layer '%s' references bottom '%s' at the %s stage however "
                        "this blob is not included at that stage. Please consider "
                        "using an include directive to limit the scope of this layer."
                        % (
                            layer.name, bottom,
                            "TRAIN" if phase == caffe_pb2.TRAIN else "TEST"
                        )
                    )


def cleanedUpClassificationNetwork(original_network, num_categories):
    """
    Perform a few cleanup routines on a classification network
    Returns a new NetParameter
    """
    network = caffe_pb2.NetParameter()
    network.CopyFrom(original_network)

    for i, layer in enumerate(network.layer):
        if 'Data' in layer.type:
            assert layer.type in ['Data', 'HDF5Data'], \
                'Unsupported data layer type %s' % layer.type

        elif layer.type == 'Input':
            # DIGITS handles the deploy file for you
            del network.layer[i]

        elif layer.type == 'Accuracy':
            # Check to see if top_k > num_categories
            if (layer.accuracy_param.HasField('top_k') and
                    layer.accuracy_param.top_k > num_categories):
                del network.layer[i]

        elif layer.type == 'InnerProduct':
            # Check to see if num_output is unset
            if not layer.inner_product_param.HasField('num_output'):
                layer.inner_product_param.num_output = num_categories

    return network


def cleanedUpGenericNetwork(original_network):
    """
    Perform a few cleanup routines on a generic network
    Returns a new NetParameter
    """
    network = caffe_pb2.NetParameter()
    network.CopyFrom(original_network)

    for i, layer in enumerate(network.layer):
        if 'Data' in layer.type:
            assert layer.type in ['Data'], \
                'Unsupported data layer type %s' % layer.type

        elif layer.type == 'Input':
            # DIGITS handles the deploy file for you
            del network.layer[i]

        elif layer.type == 'InnerProduct':
            # Check to see if num_output is unset
            assert layer.inner_product_param.HasField('num_output'), \
                "Don't leave inner_product_param.num_output unset for generic networks (layer %s)" % layer.name

    return network


def filterLayersByState(network):
    """
    Splits up a network into data, train_val and deploy layers
    """
    # The net has a NetState when in use
    train_state = caffe_pb2.NetState()
    text_format.Merge('phase: TRAIN stage: "train"', train_state)
    val_state = caffe_pb2.NetState()
    text_format.Merge('phase: TEST stage: "val"', val_state)
    deploy_state = caffe_pb2.NetState()
    text_format.Merge('phase: TEST stage: "deploy"', deploy_state)

    # Each layer can have several NetStateRules
    train_rule = caffe_pb2.NetStateRule()
    text_format.Merge('phase: TRAIN', train_rule)
    val_rule = caffe_pb2.NetStateRule()
    text_format.Merge('phase: TEST', val_rule)

    # Return three NetParameters
    data_layers = caffe_pb2.NetParameter()
    train_val_layers = caffe_pb2.NetParameter()
    deploy_layers = caffe_pb2.NetParameter()

    for layer in network.layer:
        included_train = _layerIncludedInState(layer, train_state)
        included_val = _layerIncludedInState(layer, val_state)
        included_deploy = _layerIncludedInState(layer, deploy_state)

        # Treat data layers differently (more processing done later)
        if 'Data' in layer.type:
            data_layers.layer.add().CopyFrom(layer)
            rule = None
            if not included_train:
                # Exclude from train
                rule = val_rule
            elif not included_val:
                # Exclude from val
                rule = train_rule
            _setLayerRule(data_layers.layer[-1], rule)

        # Non-data layers
        else:
            if included_train or included_val:
                # Add to train_val
                train_val_layers.layer.add().CopyFrom(layer)
                rule = None
                if not included_train:
                    # Exclude from train
                    rule = val_rule
                elif not included_val:
                    # Exclude from val
                    rule = train_rule
                _setLayerRule(train_val_layers.layer[-1], rule)

            if included_deploy:
                # Add to deploy
                deploy_layers.layer.add().CopyFrom(layer)
                _setLayerRule(deploy_layers.layer[-1], None)

    return (data_layers, train_val_layers, deploy_layers)


def _layerIncludedInState(layer, state):
    """
    Returns True if this layer will be included in the given state
    Logic copied from Caffe's Net::FilterNet()
    """
    # If no include rules are specified, the layer is included by default and
    # only excluded if it meets one of the exclude rules.
    layer_included = len(layer.include) == 0

    for exclude_rule in layer.exclude:
        if _stateMeetsRule(state, exclude_rule):
            layer_included = False
            break

    for include_rule in layer.include:
        if _stateMeetsRule(state, include_rule):
            layer_included = True
            break

    return layer_included


def _stateMeetsRule(state, rule):
    """
    Returns True if the given state meets the given rule
    Logic copied from Caffe's Net::StateMeetsRule()
    """
    if rule.HasField('phase'):
        if rule.phase != state.phase:
            return False

    if rule.HasField('min_level'):
        if state.level < rule.min_level:
            return False

    if rule.HasField('max_level'):
        if state.level > rule.max_level:
            return False

    # The state must contain ALL of the rule's stages
    for stage in rule.stage:
        if stage not in state.stage:
            return False

    # The state must contain NONE of the rule's not_stages
    for stage in rule.not_stage:
        if stage in state.stage:
            return False

    return True


def _setLayerRule(layer, rule=None):
    """
    Set a new include rule for this layer
    If rule is None, the layer will always be included
    """
    layer.ClearField('include')
    layer.ClearField('exclude')
    if rule is not None:
        layer.include.add().CopyFrom(rule)
