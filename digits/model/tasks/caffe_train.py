# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import time
import math
import subprocess
import operator

import numpy as np
import scipy
from google.protobuf import text_format
import caffe
import caffe_pb2

from train import TrainTask
from digits.config import config_value
from digits.status import Status
from digits import utils, dataset
from digits.utils import subclass, override, constants

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 3

# Constants
CAFFE_SOLVER_FILE = 'solver.prototxt'
CAFFE_TRAIN_VAL_FILE = 'train_val.prototxt'
CAFFE_SNAPSHOT_PREFIX = 'snapshot'
CAFFE_DEPLOY_FILE = 'deploy.prototxt'

@subclass
class CaffeTrainTask(TrainTask):
    """
    Trains a caffe model
    """

    CAFFE_LOG = 'caffe_output.log'

    @staticmethod
    def upgrade_network(network):
        #TODO
        pass

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
        self.train_val_file = CAFFE_TRAIN_VAL_FILE
        self.snapshot_prefix = CAFFE_SNAPSHOT_PREFIX
        self.deploy_file = CAFFE_DEPLOY_FILE
        self.log_file = self.CAFFE_LOG

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
            print 'Upgrading CaffeTrainTask to version 2 ...'
            self.caffe_log_file = self.CAFFE_LOG
        if state['pickver_task_caffe_train'] <= 2:
            print 'Upgrading CaffeTrainTask to version 3 ...'
            self.log_file = self.caffe_log_file
            self.framework_id = 'caffe'
        self.pickver_task_caffe_train = PICKLE_VERSION

        # Make changes to self
        self.loaded_snapshot_file = None
        self.loaded_snapshot_epoch = None

        # These things don't get pickled
        self.image_mean = None

    ### Task overrides

    @override
    def name(self):
        return 'Train Caffe Model'

    @override
    def before_run(self):
        super(CaffeTrainTask, self).before_run()

        if isinstance(self.dataset, dataset.ImageClassificationDatasetJob):
            self.save_files_classification()
        elif isinstance(self.dataset, dataset.GenericImageDatasetJob):
            self.save_files_generic()
        else:
            raise NotImplementedError

        self.caffe_log = open(self.path(self.CAFFE_LOG), 'a')
        self.saving_snapshot = False
        self.receiving_train_output = False
        self.receiving_val_output = False
        self.last_train_update = None
        return True

    def get_mean_image(self, mean_file, resize = False):
        mean_image = None
        with open(self.dataset.path(mean_file), 'rb') as f:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(f.read())

            if isinstance(self.dataset, dataset.ImageClassificationDatasetJob):
                mean_image = np.reshape(blob.data,
                                        (
                                            self.dataset.image_dims[2],
                                            self.dataset.image_dims[0],
                                            self.dataset.image_dims[1],
                                        )
                                    )
            elif isinstance(self.dataset, dataset.GenericImageDatasetJob):
                task = self.dataset.analyze_db_tasks()[0]
                mean_image = np.reshape(blob.data,
                                        (
                                            task.image_channels,
                                            task.image_height,
                                            task.image_width,
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
                mean_image = mean_image.transpose(1,2,0)

                shape = list(mean_image.shape)
                # imresize will not resize if the depth is anything
                # other than 3 or 4.  If it's 1, imresize expects an
                # array.
                if (len(shape) == 2 or (len(shape) == 3 and (shape[2] == 3 or shape[2] == 4))):
                    mean_image = scipy.misc.imresize(mean_image, (data_shape[2], data_shape[3]))
                else:
                    mean_image = scipy.misc.imresize(mean_image[:,:,0],
                                                     (data_shape[2], data_shape[3]))
                    mean_image = np.expand_dims(mean_image, axis=2)
                mean_image = mean_image.transpose(2,0,1)
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
    # TODO break them up into separate functions
    def save_files_classification(self):
        """
        Save solver, train_val and deploy files to disk
        """
        has_val_set = self.dataset.val_db_task() is not None

        ### Check what has been specified in self.network

        tops = []
        bottoms = {}
        train_data_layer = None
        val_data_layer = None
        hidden_layers = caffe_pb2.NetParameter()
        loss_layers = []
        accuracy_layers = []
        for layer in self.network.layer:
            assert layer.type not in ['DummyData', 'ImageData', 'MemoryData', 'WindowData'], 'unsupported data layer type'
            if layer.type in ['Data', 'HDF5Data']:
                for rule in layer.include:
                    if rule.phase == caffe_pb2.TRAIN:
                        assert train_data_layer is None, 'cannot specify two train data layers'
                        train_data_layer = layer
                    elif rule.phase == caffe_pb2.TEST:
                        assert val_data_layer is None, 'cannot specify two test data layers'
                        val_data_layer = layer
            elif layer.type == 'SoftmaxWithLoss':
                loss_layers.append(layer)
            elif layer.type == 'Accuracy':
                addThis = True
                if layer.accuracy_param.HasField('top_k'):
                    if layer.accuracy_param.top_k >= len(self.get_labels()):
                        self.logger.warning('Removing layer %s because top_k=%s while there are are only %s labels in this dataset' % (layer.name, layer.accuracy_param.top_k, len(self.get_labels())))
                        addThis = False
                if addThis:
                    accuracy_layers.append(layer)
            else:
                hidden_layers.layer.add().CopyFrom(layer)
                if len(layer.bottom) == 1 and len(layer.top) == 1 and layer.bottom[0] == layer.top[0]:
                    pass
                else:
                    for top in layer.top:
                        tops.append(top)
                    for bottom in layer.bottom:
                        bottoms[bottom] = True

        if train_data_layer is None:
            assert val_data_layer is None, 'cannot specify a test data layer without a train data layer'

        assert len(loss_layers) > 0, 'must specify a loss layer'

        network_outputs = []
        for name in tops:
            if name not in bottoms:
                network_outputs.append(name)
        assert len(network_outputs), 'network must have an output'

        # Update num_output for any output InnerProduct layers automatically
        for layer in hidden_layers.layer:
            if layer.type == 'InnerProduct':
                for top in layer.top:
                    if top in network_outputs:
                        layer.inner_product_param.num_output = len(self.get_labels())
                        break

        ### Write train_val file

        train_val_network = caffe_pb2.NetParameter()

        dataset_backend = self.dataset.train_db_task().backend

        # data layers
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
            max_crop_size = min(self.dataset.image_dims[0], self.dataset.image_dims[1])
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
            train_data_layer = train_val_network.layer.add(type = layer_type, name = 'data')
            train_data_layer.top.append('data')
            train_data_layer.top.append('label')
            train_data_layer.include.add(phase = caffe_pb2.TRAIN)
            if dataset_backend == 'lmdb':
                train_data_layer.data_param.batch_size = constants.DEFAULT_BATCH_SIZE
            elif dataset_backend == 'hdf5':
                train_data_layer.hdf5_data_param.batch_size = constants.DEFAULT_BATCH_SIZE
            if self.crop_size:
                assert dataset_backend != 'hdf5', 'HDF5Data layer does not support cropping'
                train_data_layer.transform_param.crop_size = self.crop_size
            if has_val_set:
                val_data_layer = train_val_network.layer.add(type = layer_type, name = 'data')
                val_data_layer.top.append('data')
                val_data_layer.top.append('label')
                val_data_layer.include.add(phase = caffe_pb2.TEST)
                if dataset_backend == 'lmdb':
                    val_data_layer.data_param.batch_size = constants.DEFAULT_BATCH_SIZE
                elif dataset_backend == 'hdf5':
                    val_data_layer.hdf5_data_param.batch_size = constants.DEFAULT_BATCH_SIZE
                if self.crop_size:
                    val_data_layer.transform_param.crop_size = self.crop_size
        if dataset_backend == 'lmdb':
            train_data_layer.data_param.source = self.dataset.path(self.dataset.train_db_task().db_name)
            train_data_layer.data_param.backend = caffe_pb2.DataParameter.LMDB
            if val_data_layer is not None and has_val_set:
                val_data_layer.data_param.source = self.dataset.path(self.dataset.val_db_task().db_name)
                val_data_layer.data_param.backend = caffe_pb2.DataParameter.LMDB
        elif dataset_backend == 'hdf5':
            train_data_layer.hdf5_data_param.source = self.dataset.path(self.dataset.train_db_task().textfile)
            if val_data_layer is not None and has_val_set:
                val_data_layer.hdf5_data_param.source = self.dataset.path(self.dataset.val_db_task().textfile)

        if self.use_mean == 'pixel':
            assert dataset_backend != 'hdf5', 'HDF5Data layer does not support mean subtraction'
            mean_pixel = self.get_mean_pixel(self.dataset.path(self.dataset.train_db_task().mean_file))
            self.set_mean_value(train_data_layer, mean_pixel)
            if val_data_layer is not None and has_val_set:
                self.set_mean_value(val_data_layer, mean_pixel)

        elif self.use_mean == 'image':
            self.set_mean_file(train_data_layer, self.dataset.path(self.dataset.train_db_task().mean_file))
            if val_data_layer is not None and has_val_set:
                self.set_mean_file(val_data_layer, self.dataset.path(self.dataset.train_db_task().mean_file))

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
                if val_data_layer is not None and has_val_set and not val_data_layer.hdf5_data_param.HasField('batch_size'):
                    val_data_layer.hdf5_data_param.batch_size = constants.DEFAULT_BATCH_SIZE

        # hidden layers
        train_val_network.MergeFrom(hidden_layers)

        # output layers
        train_val_network.layer.extend(loss_layers)
        train_val_network.layer.extend(accuracy_layers)

        with open(self.path(self.train_val_file), 'w') as outfile:
            text_format.PrintMessage(train_val_network, outfile)

        ### Write deploy file

        deploy_network = caffe_pb2.NetParameter()

        # input
        deploy_network.input.append('data')
        shape = deploy_network.input_shape.add()
        shape.dim.append(1)
        shape.dim.append(self.dataset.image_dims[2])
        if self.crop_size:
            shape.dim.append(self.crop_size)
            shape.dim.append(self.crop_size)
        else:
            shape.dim.append(self.dataset.image_dims[0])
            shape.dim.append(self.dataset.image_dims[1])

        # hidden layers
        deploy_network.MergeFrom(hidden_layers)

        # output layers
        if loss_layers[-1].type == 'SoftmaxWithLoss':
            prob_layer = deploy_network.layer.add(
                    type = 'Softmax',
                    name = 'prob')
            prob_layer.bottom.append(network_outputs[-1])
            prob_layer.top.append('prob')

        with open(self.path(self.deploy_file), 'w') as outfile:
            text_format.PrintMessage(deploy_network, outfile)

        ### Write solver file

        solver = caffe_pb2.SolverParameter()
        # get enum value for solver type
        solver.solver_type = getattr(solver, self.solver_type)
        solver.net = self.train_val_file

        # Set CPU/GPU mode
        if config_value('caffe_root')['cuda_enabled'] and \
                bool(config_value('gpu_list')):
            solver.solver_mode = caffe_pb2.SolverParameter.GPU
        else:
            solver.solver_mode = caffe_pb2.SolverParameter.CPU

        solver.snapshot_prefix = self.snapshot_prefix

        # Epochs -> Iterations
        train_iter = int(math.ceil(float(self.dataset.train_db_task().entries_count) / train_data_layer.data_param.batch_size))
        solver.max_iter = train_iter * self.train_epochs
        snapshot_interval = self.snapshot_interval * train_iter
        if 0 < snapshot_interval <= 1:
            solver.snapshot = 1 # don't round down
        elif 1 < snapshot_interval < solver.max_iter:
            solver.snapshot = int(snapshot_interval)
        else:
            solver.snapshot = 0 # only take one snapshot at the end

        if has_val_set and self.val_interval:
            solver.test_iter.append(int(math.ceil(float(self.dataset.val_db_task().entries_count) / val_data_layer.data_param.batch_size)))
            val_interval = self.val_interval * train_iter
            if 0 < val_interval <= 1:
                solver.test_interval = 1 # don't round down
            elif 1 < val_interval < solver.max_iter:
                solver.test_interval = int(val_interval)
            else:
                solver.test_interval = solver.max_iter # only test once at the end

        # Learning rate
        solver.base_lr = self.learning_rate
        solver.lr_policy = self.lr_policy['policy']
        scale = float(solver.max_iter)/100.0
        if solver.lr_policy == 'fixed':
            pass
        elif solver.lr_policy == 'step':
            # stepsize = stepsize * scale
            solver.stepsize = int(math.ceil(float(self.lr_policy['stepsize']) * scale))
            solver.gamma = self.lr_policy['gamma']
        elif solver.lr_policy == 'multistep':
            for value in self.lr_policy['stepvalue']:
                # stepvalue = stepvalue * scale
                solver.stepvalue.append(int(math.ceil(float(value) * scale)))
            solver.gamma = self.lr_policy['gamma']
        elif solver.lr_policy == 'exp':
            # gamma = gamma^(1/scale)
            solver.gamma = math.pow(self.lr_policy['gamma'], 1.0/scale)
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

        # go with the suggested defaults
        if solver.solver_type != solver.ADAGRAD:
            solver.momentum = 0.9
        solver.weight_decay = 0.0005

        # Display 8x per epoch, or once per 5000 images, whichever is more frequent
        solver.display = max(1, min(
                int(math.floor(float(solver.max_iter) / (self.train_epochs * 8))),
                int(math.ceil(5000.0 / train_data_layer.data_param.batch_size))
                ))

        if self.random_seed is not None:
            solver.random_seed = self.random_seed

        with open(self.path(self.solver_file), 'w') as outfile:
            text_format.PrintMessage(solver, outfile)
        self.solver = solver # save for later

        return True


    def save_files_generic(self):
        """
        Save solver, train_val and deploy files to disk
        """
        train_image_db = None
        train_labels_db = None
        val_image_db = None
        val_labels_db = None
        for task in self.dataset.tasks:
            if task.purpose == 'Training Images':
                train_image_db = task
            if task.purpose == 'Training Labels':
                train_labels_db = task
            if task.purpose == 'Validation Images':
                val_image_db = task
            if task.purpose == 'Validation Labels':
                val_labels_db = task

        assert train_image_db is not None, 'Training images are required'

        ### Split up train_val and deploy layers

        train_image_data_layer = None
        train_label_data_layer = None
        val_image_data_layer = None
        val_label_data_layer = None

        train_val_layers = caffe_pb2.NetParameter()
        deploy_layers = caffe_pb2.NetParameter()

        for layer in self.network.layer:
            assert layer.type not in ['MemoryData', 'HDF5Data', 'ImageData'], 'unsupported data layer type'
            if layer.name.startswith('train_'):
                train_val_layers.layer.add().CopyFrom(layer)
                train_val_layers.layer[-1].name = train_val_layers.layer[-1].name[6:]
            elif layer.name.startswith('deploy_'):
                deploy_layers.layer.add().CopyFrom(layer)
                deploy_layers.layer[-1].name = deploy_layers.layer[-1].name[7:]
            elif layer.type == 'Data':
                for rule in layer.include:
                    if rule.phase == caffe_pb2.TRAIN:
                        if 'data' in layer.top:
                            assert train_image_data_layer is None, 'cannot specify two train image data layers'
                            train_image_data_layer = layer
                        elif 'label' in layer.top:
                            assert train_label_data_layer is None, 'cannot specify two train label data layers'
                            train_label_data_layer = layer
                    elif rule.phase == caffe_pb2.TEST:
                        if 'data' in layer.top:
                            assert val_image_data_layer is None, 'cannot specify two val image data layers'
                            val_image_data_layer = layer
                        elif 'label' in layer.top:
                            assert val_label_data_layer is None, 'cannot specify two val label data layers'
                            val_label_data_layer = layer
            elif 'loss' in layer.type.lower():
                # Don't add it to the deploy network
                train_val_layers.layer.add().CopyFrom(layer)
            elif 'accuracy' in layer.type.lower():
                # Don't add it to the deploy network
                train_val_layers.layer.add().CopyFrom(layer)
            else:
                train_val_layers.layer.add().CopyFrom(layer)
                deploy_layers.layer.add().CopyFrom(layer)

        ### Write train_val file

        train_val_network = caffe_pb2.NetParameter()

        # data layers
        train_image_data_layer = self.make_generic_data_layer(train_image_db, train_image_data_layer, 'data', 'data', caffe_pb2.TRAIN)
        if train_image_data_layer is not None:
            train_val_network.layer.add().CopyFrom(train_image_data_layer)
        train_label_data_layer = self.make_generic_data_layer(train_labels_db, train_label_data_layer, 'label', 'label', caffe_pb2.TRAIN)
        if train_label_data_layer is not None:
            train_val_network.layer.add().CopyFrom(train_label_data_layer)

        val_image_data_layer = self.make_generic_data_layer(val_image_db, val_image_data_layer, 'data', 'data', caffe_pb2.TEST)
        if val_image_data_layer is not None:
            train_val_network.layer.add().CopyFrom(val_image_data_layer)
        val_label_data_layer = self.make_generic_data_layer(val_labels_db, val_label_data_layer, 'label', 'label', caffe_pb2.TEST)
        if val_label_data_layer is not None:
            train_val_network.layer.add().CopyFrom(val_label_data_layer)

        # hidden layers
        train_val_network.MergeFrom(train_val_layers)

        with open(self.path(self.train_val_file), 'w') as outfile:
            text_format.PrintMessage(train_val_network, outfile)

        ### Write deploy file

        deploy_network = caffe_pb2.NetParameter()

        # input
        deploy_network.input.append('data')
        shape = deploy_network.input_shape.add()
        shape.dim.append(1)
        shape.dim.append(train_image_db.image_channels)
        if train_image_data_layer.transform_param.HasField('crop_size'):
            shape.dim.append(
                    train_image_data_layer.transform_param.crop_size)
            shape.dim.append(
                    train_image_data_layer.transform_param.crop_size)
        else:
            shape.dim.append(train_image_db.image_height)
            shape.dim.append(train_image_db.image_width)

        # hidden layers
        deploy_network.MergeFrom(deploy_layers)

        with open(self.path(self.deploy_file), 'w') as outfile:
            text_format.PrintMessage(deploy_network, outfile)

        ### Write solver file

        solver = caffe_pb2.SolverParameter()
        # get enum value for solver type
        solver.solver_type = getattr(solver, self.solver_type)
        solver.net = self.train_val_file

        # Set CPU/GPU mode
        if config_value('caffe_root')['cuda_enabled'] and \
                bool(config_value('gpu_list')):
            solver.solver_mode = caffe_pb2.SolverParameter.GPU
        else:
            solver.solver_mode = caffe_pb2.SolverParameter.CPU

        solver.snapshot_prefix = self.snapshot_prefix

        # Epochs -> Iterations
        train_iter = int(math.ceil(float(train_image_db.image_count) / train_image_data_layer.data_param.batch_size))
        solver.max_iter = train_iter * self.train_epochs
        snapshot_interval = self.snapshot_interval * train_iter
        if 0 < snapshot_interval <= 1:
            solver.snapshot = 1 # don't round down
        elif 1 < snapshot_interval < solver.max_iter:
            solver.snapshot = int(snapshot_interval)
        else:
            solver.snapshot = 0 # only take one snapshot at the end

        if val_image_data_layer:
            solver.test_iter.append(int(math.ceil(float(val_image_db.image_count) / val_image_data_layer.data_param.batch_size)))
            val_interval = self.val_interval * train_iter
            if 0 < val_interval <= 1:
                solver.test_interval = 1 # don't round down
            elif 1 < val_interval < solver.max_iter:
                solver.test_interval = int(val_interval)
            else:
                solver.test_interval = solver.max_iter # only test once at the end

        # Learning rate
        solver.base_lr = self.learning_rate
        solver.lr_policy = self.lr_policy['policy']
        scale = float(solver.max_iter)/100.0
        if solver.lr_policy == 'fixed':
            pass
        elif solver.lr_policy == 'step':
            # stepsize = stepsize * scale
            solver.stepsize = int(math.ceil(float(self.lr_policy['stepsize']) * scale))
            solver.gamma = self.lr_policy['gamma']
        elif solver.lr_policy == 'multistep':
            for value in self.lr_policy['stepvalue']:
                # stepvalue = stepvalue * scale
                solver.stepvalue.append(int(math.ceil(float(value) * scale)))
            solver.gamma = self.lr_policy['gamma']
        elif solver.lr_policy == 'exp':
            # gamma = gamma^(1/scale)
            solver.gamma = math.pow(self.lr_policy['gamma'], 1.0/scale)
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

        # go with the suggested defaults
        if solver.solver_type != solver.ADAGRAD:
            solver.momentum = 0.9
        solver.weight_decay = 0.0005

        # Display 8x per epoch, or once per 5000 images, whichever is more frequent
        solver.display = max(1, min(
                int(math.floor(float(solver.max_iter) / (self.train_epochs * 8))),
                int(math.ceil(5000.0 / train_image_data_layer.data_param.batch_size))
                ))

        if self.random_seed is not None:
            solver.random_seed = self.random_seed

        with open(self.path(self.solver_file), 'w') as outfile:
            text_format.PrintMessage(solver, outfile)
        self.solver = solver # save for later

        return True

    def make_generic_data_layer(self, db, orig_layer, name, top, phase):
        """
        Utility within save_files_generic for creating a Data layer
        Returns a LayerParameter (or None)

        Arguments:
        db -- an AnalyzeDbTask (or None)
        orig_layer -- a LayerParameter supplied by the user (or None)
        """
        if db is None:
            #TODO allow user to specify a standard data layer even if it doesn't exist in the dataset
            return None
        layer = caffe_pb2.LayerParameter()
        if orig_layer is not None:
            layer.CopyFrom(orig_layer)
        layer.type = 'Data'
        layer.name = name
        if top not in layer.top:
            layer.ClearField('top')
            layer.top.append(top)
        layer.ClearField('include')
        layer.include.add(phase=phase)

        # source
        if layer.data_param.HasField('source'):
            self.logger.warning('Ignoring data_param.source ...')
        layer.data_param.source = db.path(db.database)
        if layer.data_param.HasField('backend'):
            self.logger.warning('Ignoring data_param.backend ...')
        layer.data_param.backend = caffe_pb2.DataParameter.LMDB

        # batch size
        if not layer.data_param.HasField('batch_size'):
            layer.data_param.batch_size = constants.DEFAULT_BATCH_SIZE
        if self.batch_size:
            layer.data_param.batch_size = self.batch_size

        # mean
        if name == 'data' and self.dataset.mean_file:
            if self.use_mean == 'pixel':
                mean_pixel = self.get_mean_pixel(self.dataset.path(self.dataset.mean_file))
                ## remove any values that may already be in the network
                self.set_mean_value(layer, mean_pixel)
            elif self.use_mean == 'image':
                self.set_mean_file(layer, self.dataset.path(self.dataset.mean_file))

        # crop size
        if name == 'data' and self.crop_size:
            max_crop_size = min(db.image_width, db.image_height)
            assert self.crop_size <= max_crop_size, 'crop_size is larger than the image size'
            layer.transform_param.crop_size = self.crop_size
        return layer

    def iteration_to_epoch(self, it):
        return float(it * self.train_epochs) / self.solver.max_iter

    @override
    def task_arguments(self, resources, env):
        args = [config_value('caffe_root')['executable'],
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
                if config_value('caffe_root')['version'] < utils.parse_version('0.14.0-alpha'):
                    # Prior to version 0.14, NVcaffe used the --gpus switch
                    args.append('--gpus=%s' % ','.join(identifiers))
                else:
                    args.append('--gpu=%s' % ','.join(identifiers))
        if self.pretrained_model:
            args.append('--weights=%s' % self.path(self.pretrained_model))

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
            index = int(match.group(2))
            name = match.group(3)
            value = match.group(4)
            assert value.lower() != 'nan', 'Network outputted NaN for "%s" (%s phase). Try decreasing your learning rate.' % (name, phase)
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
                self.logger.warning('caffe output format seems to have changed. Expected "Snapshotting solver state..." after "Snapshotting to..."')
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
            elif level == 'F': #FAIL
                level = 'critical'
            timestamp = time.mktime(time.strptime(timestr, '%Y%m%d %H:%M:%S'))
            return (timestamp, level, message)
        else:
            #self.logger.warning('Unrecognized task output "%s"' % line)
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
            output = subprocess.check_output(['tail', '-n40', self.path(self.CAFFE_LOG)])
            lines = []
            for line in output.split('\n'):
                # parse caffe header
                timestamp, level, message = self.preprocess_output_caffe(line)

                if message:
                    lines.append(message)
            # return the last 20 lines
            self.traceback = '\n'.join(lines[len(lines)-20:])
            if 'DIGITS_MODE_TEST' in os.environ:
                print output

    ### TrainTask overrides

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
                epoch = float(iteration) / (float(self.solver.max_iter)/self.train_epochs)
                # assert epoch.is_integer(), '%s is not an integer' % epoch
                epoch = round(epoch,3)
                # if epoch is int
                if epoch == math.ceil(epoch):
                    # print epoch,math.ceil(epoch),int(epoch)
                    epoch = int(epoch)
                snapshots.append( (
                        os.path.join(snapshot_dir, filename),
                        epoch
                        )
                    )
            # find solverstates
            match = re.match(r'%s_iter_(\d+)\.solverstate' % os.path.basename(self.snapshot_prefix), filename)
            if match:
                solverstates.append( (
                        os.path.join(snapshot_dir, filename),
                        int(match.group(1))
                        )
                    )

        # delete all but the most recent solverstate
        for filename, iteration in sorted(solverstates, key=lambda tup: tup[1])[:-1]:
            #print 'Removing "%s"' % filename
            os.remove(filename)

        self.snapshots = sorted(snapshots, key=lambda tup: tup[1])

        return len(self.snapshots) > 0

    @override
    def est_next_snapshot(self):
        if self.status != Status.RUN or self.current_iteration == 0:
            return None
        elapsed = time.time() - self.status_updates[-1][1]
        next_snapshot_iteration = (1 + self.current_iteration//self.snapshot_interval) * self.snapshot_interval
        return (next_snapshot_iteration - self.current_iteration) * elapsed // self.current_iteration

    @override
    def can_view_weights(self):
        return False

    @override
    def can_infer_one(self):
        if isinstance(self.dataset, dataset.ImageClassificationDatasetJob):
            return True
        return False

    @override
    def infer_one(self, data, snapshot_epoch=None, layers=None):
        if isinstance(self.dataset, dataset.ImageClassificationDatasetJob):
            return self.classify_one(data,
                    snapshot_epoch=snapshot_epoch,
                    layers=layers,
                    )
        elif isinstance(self.dataset, dataset.GenericImageDatasetJob):
            return self.infer_one_generic(data,
                    snapshot_epoch=snapshot_epoch,
                    layers=layers,
                    )
        raise NotImplementedError()

    def classify_one(self, image, snapshot_epoch=None, layers=None):
        """
        Classify an image
        Returns (predictions, visualizations)
            predictions -- an array of [ (label, confidence), ...] for each label, sorted by confidence
            visualizations -- a list of dicts for the specified layers
        Returns (None, None) if something goes wrong

        Arguments:
        image -- a np.array

        Keyword arguments:
        snapshot_epoch -- which snapshot to use
        layers -- which layer activation[s] and weight[s] to visualize
        """
        labels = self.get_labels()
        net = self.get_net(snapshot_epoch)

        # process image
        if image.ndim == 2:
            image = image[:,:,np.newaxis]
        preprocessed = self.get_transformer().preprocess(
                'data', image)

        # reshape net input (if necessary)
        test_shape = (1,) + preprocessed.shape
        if net.blobs['data'].data.shape != test_shape:
            net.blobs['data'].reshape(*test_shape)

        # run inference
        net.blobs['data'].data[...] = preprocessed
        output = net.forward()
        scores = output[net.outputs[-1]].flatten()
        indices = (-scores).argsort()
        predictions = []
        for i in indices:
            predictions.append( (labels[i], scores[i]) )

        visualizations = self.get_layer_visualizations(net, layers)
        return (predictions, visualizations)

    def infer_one_generic(self, image, snapshot_epoch=None, layers=None):
        """
        Run inference on one image for a generic model
        Returns (output, visualizations)
            output -- an dict of string -> np.ndarray
            visualizations -- a list of dicts for the specified layers
        Returns (None, None) if something goes wrong

        Arguments:
        image -- an np.ndarray

        Keyword arguments:
        snapshot_epoch -- which snapshot to use
        layers -- which layer activation[s] and weight[s] to visualize
        """
        net = self.get_net(snapshot_epoch)

        # process image
        if image.ndim == 2:
            image = image[:,:,np.newaxis]
        preprocessed = self.get_transformer().preprocess(
                'data', image)

        # reshape net input (if necessary)
        test_shape = (1,) + preprocessed.shape
        if net.blobs['data'].data.shape != test_shape:
            net.blobs['data'].reshape(*test_shape)

        # run inference
        net.blobs['data'].data[...] = preprocessed
        output = net.forward()

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
                    print 'Computing visualizations for "%s" ...' % layer.name
                    for bottom in layer.bottom:
                        if bottom in net.blobs and bottom not in added_activations:
                            data = net.blobs[bottom].data[0]
                            vis = utils.image.get_layer_vis_square(data,
                                    allow_heatmap=bool(bottom != 'data'))
                            mean, std, hist = self.get_layer_statistics(data)
                            visualizations.append(
                                    {
                                        'name': str(bottom),
                                        'vis_type': 'Activation',
                                        'image_html': utils.image.embed_image_html(vis),
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
                            vis = utils.image.get_layer_vis_square(data)
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
                                    'image_html': utils.image.embed_image_html(vis),
                                    'data_stats': {
                                        'shape':data.shape,
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
                                    normalize = normalize,
                                    allow_heatmap = bool(top != 'data'))
                            mean, std, hist = self.get_layer_statistics(data)
                            visualizations.append(
                                    {
                                        'name': str(top),
                                        'vis_type': 'Activation',
                                        'image_html': utils.image.embed_image_html(vis),
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
        mean = np.mean(data)
        std = np.std(data)
        y, x = np.histogram(data, bins=20)
        y = list(y)
        ticks = x[[0,len(x)/2,-1]]
        x = [(x[i]+x[i+1])/2.0 for i in xrange(len(x)-1)]
        ticks = list(ticks)
        return (mean, std, [y, x, ticks])

    @override
    def can_infer_many(self):
        if isinstance(self.dataset, dataset.ImageClassificationDatasetJob):
            return True
        return False

    @override
    def infer_many(self, data, snapshot_epoch=None):
        if isinstance(self.dataset, dataset.ImageClassificationDatasetJob):
            return self.classify_many(data, snapshot_epoch=snapshot_epoch)
        elif isinstance(self.dataset, dataset.GenericImageDatasetJob):
            return self.infer_many_generic(data, snapshot_epoch=snapshot_epoch)
        raise NotImplementedError()

    def classify_many(self, images, snapshot_epoch=None):
        """
        Returns (labels, results):
        labels -- an array of strings
        results -- a 2D np array:
            [
                [image0_label0_confidence, image0_label1_confidence, ...],
                [image1_label0_confidence, image1_label1_confidence, ...],
                ...
            ]

        Arguments:
        images -- a list of np.arrays

        Keyword arguments:
        snapshot_epoch -- which snapshot to use
        """
        labels = self.get_labels()
        net = self.get_net(snapshot_epoch)

        caffe_images = []
        for image in images:
            if image.ndim == 2:
                caffe_images.append(image[:,:,np.newaxis])
            else:
                caffe_images.append(image)

        caffe_images = np.array(caffe_images)

        data_shape = tuple(self.get_transformer().inputs['data'])[1:]

        if self.batch_size:
            data_shape = (self.batch_size,) + data_shape
        # TODO: grab batch_size from the TEST phase in train_val network
        else:
            data_shape = (constants.DEFAULT_BATCH_SIZE,) + data_shape

        scores = None
        for chunk in [caffe_images[x:x+data_shape[0]] for x in xrange(0, len(caffe_images), data_shape[0])]:
            new_shape = (len(chunk),) + data_shape[1:]
            if net.blobs['data'].data.shape != new_shape:
                net.blobs['data'].reshape(*new_shape)
            for index, image in enumerate(chunk):
                net.blobs['data'].data[index] = self.get_transformer().preprocess(
                        'data', image)
            output = net.forward()[net.outputs[-1]]
            if scores is None:
                scores = np.copy(output)
            else:
                scores = np.vstack((scores, output))
            print 'Processed %s/%s images' % (len(scores), len(caffe_images))

        return (labels, scores)

    def infer_many_generic(self, images, snapshot_epoch=None):
        """
        Returns a list of np.ndarrays, one for each image

        Arguments:
        images -- a list of np.arrays

        Keyword arguments:
        snapshot_epoch -- which snapshot to use
        """
        net = self.get_net(snapshot_epoch)

        caffe_images = []
        for image in images:
            if image.ndim == 2:
                caffe_images.append(image[:,:,np.newaxis])
            else:
                caffe_images.append(image)

        caffe_images = np.array(caffe_images)

        data_shape = tuple(self.get_transformer().inputs['data'])[1:]

        if self.batch_size:
            data_shape = (self.batch_size,) + data_shape
        # TODO: grab batch_size from the TEST phase in train_val network
        else:
            data_shape = (constants.DEFAULT_BATCH_SIZE,) + data_shape

        outputs = None
        for chunk in [caffe_images[x:x+data_shape[0]] for x in xrange(0, len(caffe_images), data_shape[0])]:
            new_shape = (len(chunk),) + data_shape[1:]
            if net.blobs['data'].data.shape != new_shape:
                net.blobs['data'].reshape(*new_shape)
            for index, image in enumerate(chunk):
                net.blobs['data'].data[index] = self.get_transformer().preprocess(
                        'data', image)
            o = net.forward()
            if outputs is None:
                outputs = o
            else:
                for name,blob in o.iteritems():
                    outputs[name] = np.vstack((outputs[name], blob))
            print 'Processed %s/%s images' % (len(outputs[outputs.keys()[0]]), len(caffe_images))

        return outputs

    def has_model(self):
        """
        Returns True if there is a model that can be used
        """
        return len(self.snapshots) > 0

    def get_net(self, epoch=None):
        """
        Returns an instance of caffe.Net

        Keyword Arguments:
        epoch -- which snapshot to load (default is -1 to load the most recently generated snapshot)
        """
        if not self.has_model():
            return False

        file_to_load = None

        if not epoch:
            epoch = self.snapshots[-1][1]
            file_to_load = self.snapshots[-1][0]
        else:
            for snapshot_file, snapshot_epoch in self.snapshots:
                if snapshot_epoch == epoch:
                    file_to_load = snapshot_file
                    break
        if file_to_load is None:
            raise Exception('snapshot not found for epoch "%s"' % epoch)

        # check if already loaded
        if self.loaded_snapshot_file and self.loaded_snapshot_file == file_to_load \
                and hasattr(self, '_caffe_net') and self._caffe_net is not None:
            return self._caffe_net

        if config_value('caffe_root')['cuda_enabled'] and\
                config_value('gpu_list'):
            caffe.set_mode_gpu()

        # load a new model
        self._caffe_net = caffe.Net(
                self.path(self.deploy_file),
                file_to_load,
                caffe.TEST)

        self.loaded_snapshot_epoch = epoch
        self.loaded_snapshot_file = file_to_load

        return self._caffe_net

    def get_transformer(self):
        """
        Returns an instance of caffe.io.Transformer
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

        if isinstance(self.dataset, dataset.ImageClassificationDatasetJob):
            if self.dataset.image_dims[2] == 3 and \
                    self.dataset.train_db_task().image_channel_order == 'BGR':
                # XXX see issue #59
                channel_swap = (2,1,0)

            if self.use_mean == 'pixel':
                mean_pixel = self.get_mean_pixel(self.dataset.path(self.dataset.train_db_task().mean_file))
            elif self.use_mean == 'image':
                mean_image = self.get_mean_image(self.dataset.path(self.dataset.train_db_task().mean_file), True)

        elif isinstance(self.dataset, dataset.GenericImageDatasetJob):
            task = self.dataset.analyze_db_tasks()[0]

            if task.image_channels == 3:
                # XXX see issue #59
                channel_swap = (2,1,0)

            if self.dataset.mean_file:
                if self.use_mean == 'pixel':
                    mean_pixel = self.get_mean_pixel(self.dataset.path(self.dataset.mean_file))
                elif self.use_mean == 'image':
                    mean_image = self.get_mean_image(self.dataset.path(self.dataset.mean_file), True)

        t = caffe.io.Transformer(
                inputs = {'data': tuple(data_shape)}
                )

        # transpose to (channels, height, width)
        t.set_transpose('data', (2,0,1))

        if channel_swap is not None:
            # swap color channels
            t.set_channel_swap('data', channel_swap)

        # set mean
        if self.use_mean == 'pixel' and mean_pixel is not None:
            t.set_mean('data', mean_pixel)
        elif self.use_mean == 'image' and mean_image is not None:
            t.set_mean('data', mean_image)

        #t.set_raw_scale('data', 255) # [0,255] range instead of [0,1]

        self._transformer = t
        return self._transformer

    @override
    def get_model_files(self):
        """
        return paths to model files
        """
        return {
                "Solver": self.solver_file,
                "Network (train/val)": self.train_val_file,
                "Network (deploy)": self.deploy_file
            }

    @override
    def get_network_desc(self):
        """
        return text description of model
        """
        return text_format.MessageToString(self.network)
