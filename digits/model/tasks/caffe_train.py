# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import time
import math
import subprocess

import numpy as np
from google.protobuf import text_format
import caffe
try:
    import caffe_pb2
except ImportError:
    # See issue #32
    from caffe.proto import caffe_pb2

from train import TrainTask
from digits.config import config_value
from digits.status import Status
from digits import utils, dataset
from digits.utils import subclass, override, constants
from digits.dataset import ImageClassificationDatasetJob

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 2

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

    def __init__(self, network, **kwargs):
        """
        Arguments:
        network -- a caffe NetParameter defining the network
        """
        super(CaffeTrainTask, self).__init__(**kwargs)
        self.pickver_task_caffe_train = PICKLE_VERSION

        self.network = network

        self.current_iteration = 0

        self.loaded_snapshot_file = None
        self.loaded_snapshot_epoch = None
        self.image_mean = None
        self.solver = None

        self.solver_file = constants.CAFFE_SOLVER_FILE
        self.train_val_file = constants.CAFFE_TRAIN_VAL_FILE
        self.snapshot_prefix = constants.CAFFE_SNAPSHOT_PREFIX
        self.deploy_file = constants.CAFFE_DEPLOY_FILE
        self.caffe_log_file = self.CAFFE_LOG

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
        if state['pickver_task_caffe_train'] == 1:
            print 'upgrading %s' % self.job_id
            self.caffe_log_file = self.CAFFE_LOG
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
            self.save_prototxt_files()
        else:
            raise NotImplementedError

        self.caffe_log = open(self.path(self.CAFFE_LOG), 'a')
        self.saving_snapshot = False
        self.receiving_train_output = False
        self.receiving_val_output = False
        self.last_train_update = None
        return True

    def save_prototxt_files(self):
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
            assert layer.type not in ['MemoryData', 'HDF5Data', 'ImageData'], 'unsupported data layer type'
            if layer.type == 'Data':
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

        # data layers
        if train_data_layer is not None:
            if train_data_layer.HasField('data_param'):
                assert not train_data_layer.data_param.HasField('source'), "don't set the data_param.source"
                assert not train_data_layer.data_param.HasField('backend'), "don't set the data_param.backend"
            max_crop_size = min(self.dataset.image_dims[0], self.dataset.image_dims[1])
            if self.crop_size:
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
                if val_data_layer.HasField('data_param'):
                    assert not val_data_layer.data_param.HasField('source'), "don't set the data_param.source"
                    assert not val_data_layer.data_param.HasField('backend'), "don't set the data_param.backend"
                if self.crop_size:
                    # use our error checking from the train layer
                    val_data_layer.transform_param.crop_size = self.crop_size
                train_val_network.layer.add().CopyFrom(val_data_layer)
                val_data_layer = train_val_network.layer[-1]
        else:
            train_data_layer = train_val_network.layer.add(type = 'Data', name = 'data')
            train_data_layer.top.append('data')
            train_data_layer.top.append('label')
            train_data_layer.include.add(phase = caffe_pb2.TRAIN)
            train_data_layer.data_param.batch_size = constants.DEFAULT_BATCH_SIZE
            if self.crop_size:
                train_data_layer.transform_param.crop_size = self.crop_size
            if has_val_set:
                val_data_layer = train_val_network.layer.add(type = 'Data', name = 'data')
                val_data_layer.top.append('data')
                val_data_layer.top.append('label')
                val_data_layer.include.add(phase = caffe_pb2.TEST)
                val_data_layer.data_param.batch_size = constants.DEFAULT_BATCH_SIZE
                if self.crop_size:
                    val_data_layer.transform_param.crop_size = self.crop_size
        train_data_layer.data_param.source = self.dataset.path(self.dataset.train_db_task().db_name)
        train_data_layer.data_param.backend = caffe_pb2.DataParameter.LMDB
        if val_data_layer is not None and has_val_set:
            val_data_layer.data_param.source = self.dataset.path(self.dataset.val_db_task().db_name)
            val_data_layer.data_param.backend = caffe_pb2.DataParameter.LMDB
        if self.use_mean:
            mean_pixel = None
            with open(self.dataset.path(self.dataset.train_db_task().mean_file)) as f:
                blob = caffe_pb2.BlobProto()
                blob.MergeFromString(f.read())
                mean = np.reshape(blob.data,
                        (
                            self.dataset.image_dims[2],
                            self.dataset.image_dims[0],
                            self.dataset.image_dims[1],
                            )
                        )
                mean_pixel = mean.mean(1).mean(1)
            for value in mean_pixel:
                train_data_layer.transform_param.mean_value.append(value)
            if val_data_layer is not None and has_val_set:
                for value in mean_pixel:
                    val_data_layer.transform_param.mean_value.append(value)
        if self.batch_size:
            train_data_layer.data_param.batch_size = self.batch_size
            if val_data_layer is not None and has_val_set:
                val_data_layer.data_param.batch_size = self.batch_size
        else:
            if not train_data_layer.data_param.HasField('batch_size'):
                train_data_layer.data_param.batch_size = constants.DEFAULT_BATCH_SIZE
            if val_data_layer is not None and has_val_set and not val_data_layer.data_param.HasField('batch_size'):
                val_data_layer.data_param.batch_size = constants.DEFAULT_BATCH_SIZE

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
        deploy_network.input_dim.append(1)
        deploy_network.input_dim.append(self.dataset.image_dims[2])
        if self.crop_size:
            deploy_network.input_dim.append(self.crop_size)
            deploy_network.input_dim.append(self.crop_size)
        else:
            deploy_network.input_dim.append(self.dataset.image_dims[0])
            deploy_network.input_dim.append(self.dataset.image_dims[1])

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


    def iteration_to_epoch(self, it):
        return float(it * self.train_epochs) / self.solver.max_iter

    @override
    def task_arguments(self, resources):
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
                args.append('--gpus=%s' % ','.join(identifiers))
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

        # memory requirement
        match = re.match(r'Memory required for data:\s+(\d+)', message)
        if match:
            bytes_required = int(match.group(1))
            #self.logger.debug('memory required: %s' % utils.sizeof_fmt(bytes_required))
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
        if isinstance(self.dataset, ImageClassificationDatasetJob):
            return True
        return False

    @override
    def infer_one(self, data, snapshot_epoch=None, layers=None):
        if isinstance(self.dataset, ImageClassificationDatasetJob):
            return self.classify_one(data,
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


        # add visualizations
        visualizations = []
        if layers and layers != 'none':
            if layers == 'all':
                added_activations = []
                for layer in self.network.layer:
                    print 'Computing visualizations for "%s"...' % layer.name
                    if not layer.type.endswith(('Data', 'Loss', 'Accuracy')):
                        for bottom in layer.bottom:
                            if bottom in net.blobs and bottom not in added_activations:
                                data = net.blobs[bottom].data[0]
                                vis = self.get_layer_visualization(data)
                                mean, std, hist = self.get_layer_statistics(data)
                                visualizations.append(
                                        {
                                            'name': str(bottom),
                                            'type': 'Activations',
                                            'mean': mean,
                                            'stddev': std,
                                            'histogram': hist,
                                            'image_html': utils.image.embed_image_html(vis),
                                            }
                                        )
                                added_activations.append(bottom)
                        if layer.name in net.params:
                            data = net.params[layer.name][0].data
                            if layer.type not in ['InnerProduct']:
                                vis = self.get_layer_visualization(data)
                            else:
                                vis = None
                            mean, std, hist = self.get_layer_statistics(data)
                            visualizations.append(
                                    {
                                        'name': str(layer.name),
                                        'type': 'Weights (%s layer)' % layer.type,
                                        'mean': mean,
                                        'stddev': std,
                                        'histogram': hist,
                                        'image_html': utils.image.embed_image_html(vis),
                                        }
                                    )
                        for top in layer.top:
                            if top in net.blobs and top not in added_activations:
                                data = net.blobs[top].data[0]
                                normalize = True
                                # don't normalize softmax layers
                                if layer.type == 'Softmax':
                                    normalize = False
                                vis = self.get_layer_visualization(data, normalize=normalize)
                                mean, std, hist = self.get_layer_statistics(data)
                                visualizations.append(
                                        {
                                            'name': str(top),
                                            'type': 'Activation',
                                            'mean': mean,
                                            'stddev': std,
                                            'histogram': hist,
                                            'image_html': utils.image.embed_image_html(vis),
                                            }
                                        )
                                added_activations.append(top)
            else:
                raise NotImplementedError

        return (predictions, visualizations)

    def get_layer_visualization(self, data,
            normalize = True,
            max_width = 600,
            ):
        """
        Returns a vis_square for the given layer data

        Arguments:
        data -- a np.ndarray

        Keyword arguments:
        normalize -- whether to normalize the data when visualizing
        max_width -- maximum width for the vis_square
        """
        #print 'data.shape is %s' % (data.shape,)

        if data.ndim == 1:
            # interpret as 1x1 grayscale images
            # (N, 1, 1)
            data = data[:, np.newaxis, np.newaxis]
        elif data.ndim == 2:
            # interpret as 1x1 grayscale images
            # (N, 1, 1)
            data = data.reshape((data.shape[0]*data.shape[1], 1, 1))
        elif data.ndim == 3:
            if data.shape[0] == 3:
                # interpret as a color image
                # (1, H, W,3)
                data = data[[2,1,0],...] # BGR to RGB (see issue #59)
                data = data.transpose(1,2,0)
                data = data[np.newaxis,...]
            else:
                # interpret as grayscale images
                # (N, H, W)
                pass
        elif data.ndim == 4:
            if data.shape[0] == 3:
                # interpret as HxW color images
                # (N, H, W, 3)
                data = data.transpose(1,2,3,0)
                data = data[:,:,:,[2,1,0]] # BGR to RGB (see issue #59)
            elif data.shape[1] == 3:
                # interpret as HxW color images
                # (N, H, W, 3)
                data = data.transpose(0,2,3,1)
                data = data[:,:,:,[2,1,0]] # BGR to RGB (see issue #59)
            else:
                # interpret as HxW grayscale images
                # (N, H, W)
                data = data.reshape((data.shape[0]*data.shape[1], data.shape[2], data.shape[3]))
        else:
            raise RuntimeError('unrecognized data shape: %s' % (data.shape,))

        # chop off data so that it will fit within max_width
        padsize = 0
        width = data.shape[2]
        if width > max_width:
            data = data[0,:max_width,:max_width]
        else:
            if width > 1:
                padsize = 1
                width += 1
            n = max_width/width
            n *= n
            data = data[:n]

        #print 'data.shape now %s' % (data.shape,)
        return utils.image.vis_square(data,
                padsize     = padsize,
                normalize   = normalize,
                )

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
        if isinstance(self.dataset, ImageClassificationDatasetJob):
            return True
        return False

    @override
    def infer_many(self, data, snapshot_epoch=None):
        if isinstance(self.dataset, ImageClassificationDatasetJob):
            return self.classify_many(data, snapshot_epoch=snapshot_epoch)
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

        if self.batch_size:
            data_shape = (self.batch_size, self.dataset.image_dims[2])
        # TODO: grab batch_size from the TEST phase in train_val network
        else:
            data_shape = (constants.DEFAULT_BATCH_SIZE, self.dataset.image_dims[2])

        if self.crop_size:
            data_shape += (self.crop_size, self.crop_size)
        else:
            data_shape += (self.dataset.image_dims[0], self.dataset.image_dims[1])

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
                scores = output
            else:
                scores = np.vstack((scores, output))
            print 'Processed %s/%s images' % (len(scores), len(caffe_images))

        return (labels, scores)

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

        data_shape = (1, self.dataset.image_dims[2])
        if self.crop_size:
            data_shape += (self.crop_size, self.crop_size)
        else:
            data_shape += (self.dataset.image_dims[0], self.dataset.image_dims[1])

        t = caffe.io.Transformer(
                inputs = {'data':  data_shape}
                )
        t.set_transpose('data', (2,0,1)) # transpose to (channels, height, width)

        if self.dataset.image_dims[2] == 3 and \
                self.dataset.train_db_task().image_channel_order == 'BGR':
            # channel swap
            # XXX see issue #59
            t.set_channel_swap('data', (2,1,0))

        if self.use_mean:
            # set mean
            with open(self.dataset.path(self.dataset.train_db_task().mean_file)) as f:
                blob = caffe_pb2.BlobProto()
                blob.MergeFromString(f.read())
                pixel = np.reshape(blob.data,
                        (
                            self.dataset.image_dims[2],
                            self.dataset.image_dims[0],
                            self.dataset.image_dims[1],
                            )
                        ).mean(1).mean(1)
                t.set_mean('data', pixel)

        #t.set_raw_scale('data', 255) # [0,255] range instead of [0,1]

        self._transformer = t
        return self._transformer

