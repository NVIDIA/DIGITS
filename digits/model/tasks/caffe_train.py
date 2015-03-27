# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import caffe
import time
import math
import subprocess

import numpy as np
from scipy.ndimage.interpolation import zoom
from google.protobuf import text_format
from caffe.proto import caffe_pb2

from train import TrainTask
from digits.config import config_option
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
    def upgrade_network(cls, network):
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
        self.classifier = None
        self.solver = None

        self.solver_file = constants.CAFFE_SOLVER_FILE
        self.train_val_file = constants.CAFFE_TRAIN_VAL_FILE
        self.snapshot_prefix = constants.CAFFE_SNAPSHOT_PREFIX
        self.deploy_file = constants.CAFFE_DEPLOY_FILE
        self.caffe_log_file = self.CAFFE_LOG

    def __getstate__(self):
        state = super(CaffeTrainTask, self).__getstate__()

        # Don't pickle these things
        if 'image_mean' in state:
            del state['image_mean']
        if 'classifier' in state:
            del state['classifier']
        if 'caffe_log' in state:
            del state['caffe_log']

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
        self.classifier = None

    ### Task overrides

    @override
    def name(self):
        return 'Train Caffe Model'

    @override
    def before_run(self):
        if isinstance(self.dataset, dataset.ImageClassificationDatasetJob):
            assert self.read_labels(), 'could not read labels'
            self.save_prototxt_files()
        else:
            raise NotImplementedError()

        self.caffe_log = open(self.path(self.CAFFE_LOG), 'a')
        self.saving_snapshot = False
        self.last_unimportant_update = None
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
        loss_layer = None
        accuracy_layer = None
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
                assert loss_layer is None, 'cannot specify two loss layers'
                loss_layer = layer
            elif layer.type == 'Accuracy':
                assert accuracy_layer is None, 'cannot specify two accuracy layers'
                accuracy_layer = layer
            else:
                hidden_layers.layer.add().CopyFrom(layer)
                if len(layer.bottom) == 1 and len(layer.top) == 1 and layer.bottom[0] == layer.top[0]:
                    pass
                else:
                    for top in layer.top:
                        tops.append(top)
                    for bottom in layer.bottom:
                        bottoms[bottom] = True

        assert loss_layer is not None, 'must specify a SoftmaxWithLoss layer'
        assert accuracy_layer is not None, 'must specify an Accuracy layer'
        if not has_val_set:
            self.logger.warning('Discarding Data layer for validation')
            val_data_layer = None

        output_name = None
        for name in tops:
            if name not in bottoms:
                assert output_name is None, 'network cannot have more than one output'
                output_name = name
        assert output_name is not None, 'network must have one output'
        for layer in hidden_layers.layer:
            if output_name in layer.top and layer.type == 'InnerProduct':
                layer.inner_product_param.num_output = len(self.labels)
                break

        if train_data_layer is None:
            assert val_data_layer is None, 'cannot specify a test data layer without a train data layer'

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
        if val_data_layer is not None:
            val_data_layer.data_param.source = self.dataset.path(self.dataset.val_db_task().db_name)
            val_data_layer.data_param.backend = caffe_pb2.DataParameter.LMDB
        if self.use_mean:
            train_data_layer.transform_param.mean_file = self.dataset.path(self.dataset.train_db_task().mean_file)
            if val_data_layer is not None:
                val_data_layer.transform_param.mean_file = self.dataset.path(self.dataset.train_db_task().mean_file)
        if self.batch_size:
            train_data_layer.data_param.batch_size = self.batch_size
            if val_data_layer is not None:
                val_data_layer.data_param.batch_size = self.batch_size
        else:
            if not train_data_layer.data_param.HasField('batch_size'):
                train_data_layer.data_param.batch_size = constants.DEFAULT_BATCH_SIZE
            if val_data_layer is not None and not val_data_layer.data_param.HasField('batch_size'):
                val_data_layer.data_param.batch_size = constants.DEFAULT_BATCH_SIZE

        # hidden layers
        train_val_network.MergeFrom(hidden_layers)

        # output layers
        if loss_layer is not None:
            train_val_network.layer.add().CopyFrom(loss_layer)
            loss_layer = train_val_network.layer[-1]
        else:
            loss_layer = train_val_network.layer.add(
                type = 'SoftmaxWithLoss',
                name = 'loss')
            loss_layer.bottom.append(output_name)
            loss_layer.bottom.append('label')
            loss_layer.top.append('loss')

        if accuracy_layer is not None:
            train_val_network.layer.add().CopyFrom(accuracy_layer)
            accuracy_layer = train_val_network.layer[-1]
        elif self.dataset.val_db_task():
            accuracy_layer = train_val_network.layer.add(
                    type = 'Accuracy',
                    name = 'accuracy')
            accuracy_layer.bottom.append(output_name)
            accuracy_layer.bottom.append('label')
            accuracy_layer.top.append('accuracy')
            accuracy_layer.include.add(phase = caffe_pb2.TEST)

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
        prob_layer = deploy_network.layer.add(
                type = 'Softmax',
                name = 'prob')
        prob_layer.bottom.append(output_name)
        prob_layer.top.append('prob')

        with open(self.path(self.deploy_file), 'w') as outfile:
            text_format.PrintMessage(deploy_network, outfile)

        ### Write solver file

        solver = caffe_pb2.SolverParameter()
        solver.net = self.train_val_file
        # TODO: detect if caffe is built with CPU_ONLY
        gpu_list = config_option('gpu_list')
        if gpu_list and gpu_list != 'NONE':
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

        if self.dataset.val_db_task() and self.val_interval:
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
        solver.momentum = 0.9
        solver.weight_decay = 0.0005

        # Display 8x per epoch, or once per 5000 images, whichever is more frequent
        solver.display = min(
                int(math.floor(float(solver.max_iter) / (self.train_epochs * 8))),
                int(math.ceil(5000.0 / train_data_layer.data_param.batch_size))
                )

        with open(self.path(self.solver_file), 'w') as outfile:
            text_format.PrintMessage(solver, outfile)
        self.solver = solver # save for later

        return True


    def iteration_to_epoch(self, it):
        return float(it * self.train_epochs) / self.solver.max_iter

    @override
    def task_arguments(self, **kwargs):
        gpu_id = kwargs.pop('gpu_id', None)

        if config_option('caffe_root') == 'SYS':
            caffe_bin = 'caffe'
        else:
            #caffe_bin = os.path.join(config_option('caffe_root'), 'bin', 'caffe.bin')
            caffe_bin = os.path.join(config_option('caffe_root'), 'build', 'tools', 'caffe.bin')
        args = [caffe_bin,
                'train',
                '--solver=%s' % self.path(self.solver_file),
                ]

        if gpu_id:
            args.append('--gpu=%d' % gpu_id)
        if self.pretrained_model:
            args.append('--weights=%s' % self.path(self.pretrained_model))

        return args

    @override
    def process_output(self, line):
        from digits.webapp import socketio

        self.caffe_log.write('%s\n' % line)
        self.caffe_log.flush()

        # parse caffe header
        timestamp, level, message = self.preprocess_output_caffe(line)

        if not message:
            return True

        float_exp = '(NaN|[-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?)'

        # snapshot saved
        if self.saving_snapshot:
            self.logger.info('Snapshot saved.')
            self.detect_snapshots()
            self.send_snapshot_update()
            self.saving_snapshot = False
            return True

        # loss updates
        match = re.match(r'Iteration (\d+), \w*loss\w* = %s' % float_exp, message)
        if match:
            i = int(match.group(1))
            l = match.group(2)
            assert l.lower() != 'nan', 'Network reported NaN for training loss. Try decreasing your learning rate.'
            l = float(l)
            self.train_loss_updates.append((self.iteration_to_epoch(i), l))
            self.logger.debug('Iteration %d/%d, loss=%s' % (i, self.solver.max_iter, l))
            self.send_iteration_update(i)
            self.send_data_update()
            return True

        # learning rate updates
        match = re.match(r'Iteration (\d+), lr = %s' % float_exp, message)
        if match:
            i = int(match.group(1))
            lr = match.group(2)
            if lr.lower() != 'nan':
                lr = float(lr)
                self.lr_updates.append((self.iteration_to_epoch(i), lr))
            self.send_iteration_update(i)
            return True

        # other iteration updates
        match = re.match(r'Iteration (\d+)', message)
        if match:
            i = int(match.group(1))
            self.send_iteration_update(i)
            return True

        # testing loss updates
        match = re.match(r'Test net output #\d+: \w*loss\w* = %s' % float_exp, message, flags=re.IGNORECASE)
        if match:
            l = match.group(1)
            if l.lower() != 'nan':
                l = float(l)
                self.val_loss_updates.append( (self.iteration_to_epoch(self.current_iteration), l) )
                self.send_data_update()
            return True

        # testing accuracy updates
        match = re.match(r'Test net output #(\d+): \w*acc\w* = %s' % float_exp, message, flags=re.IGNORECASE)
        if match:
            index = int(match.group(1))
            a = match.group(2)
            if a.lower() != 'nan':
                a = float(a) * 100
                self.logger.debug('Network accuracy #%d: %s' % (index, a))
                self.val_accuracy_updates.append( (self.iteration_to_epoch(self.current_iteration), a, index) )
                self.send_data_update(important=True)
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
            self.logger.debug('memory required: %s' % utils.sizeof_fmt(bytes_required))
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

    def send_iteration_update(self, it):
        """
        Sends socketio message about the current iteration
        """
        from digits.webapp import socketio

        if self.current_iteration == it:
            return

        self.current_iteration = it
        self.progress = float(it)/self.solver.max_iter

        socketio.emit('task update',
                {
                    'task': self.html_id(),
                    'update': 'progress',
                    'percentage': int(round(100*self.progress)),
                    'eta': utils.time_filters.print_time_diff(self.est_done()),
                    },
                namespace='/jobs',
                room=self.job_id,
                )

    def send_data_update(self, important=False):
        """
        Send socketio updates with the latest graph data

        Keyword arguments:
        important -- if False, only send this update if the last unimportant update was sent more than 5 seconds ago
        """
        from digits.webapp import socketio

        if not important:
            if self.last_unimportant_update and (time.time() - self.last_unimportant_update) < 5:
                return
            self.last_unimportant_update = time.time()

        # loss graph data
        data = self.loss_graph_data()
        if data:
            socketio.emit('task update',
                    {
                        'task': self.html_id(),
                        'update': 'loss_graph',
                        'data': data,
                        },
                    namespace='/jobs',
                    room=self.job_id,
                    )

        # lr graph data
        data = self.lr_graph_data()
        if data:
            socketio.emit('task update',
                    {
                        'task': self.html_id(),
                        'update': 'lr_graph',
                        'data': data,
                        },
                    namespace='/jobs',
                    room=self.job_id,
                    )

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
            visualizations -- an array of (layer_name, activations, weights) for the specified layers
        Returns (None, None) if something goes wrong

        Arguments:
        image -- a np.array

        Keyword arguments:
        snapshot_epoch -- which snapshot to use
        layers -- which layer activation[s] and weight[s] to visualize
        """
        if not self.load_model(snapshot_epoch):
            return (None, None)
        if not self.read_labels():
            return (None, None)

        # Convert to float32 in [0,1] range
        caffe_image = np.array(image).astype('float32')/255.0
        if caffe_image.ndim == 2:
            caffe_image = caffe_image[:,:,np.newaxis]

        scores = self.classifier.predict([caffe_image], oversample=False).flatten()
        indices = (-scores).argsort()
        predictions = []
        for i in indices:
            predictions.append( (self.labels[i], scores[i]) )

        visualizations = []
        if layers and layers != 'none':
            if layers == 'all':
                for layer in self.network.layer:
                    if not layer.type.endswith(('Data', 'Loss', 'Accuracy')):
                        a, w = self.get_layer_visualization(layer)
                        if a is not None or w is not None:
                            visualizations.append( (layer.name, a, w) )
            else:
                found = False
                for layer in self.network.layer:
                    if layer.name == layers:
                        a, w = self.get_layer_visualization(layer)
                        if a is not None or w is not None:
                            visualizations.append( (layer.name, a, w) )
                        found = True
                        break
                if not found:
                    raise Exception('layer does not exist: "%s"' % layers)

        return (predictions, visualizations)

    def get_layer_visualization(self, layer,
            max_width=500,
            ):
        """
        Returns (activations, params) for the given layer:
            activations -- a vis_square for the activation blobs
            weights -- a vis_square for the learned weights (may be None for some layer types)
        Returns (None, None) if an error occurs

        Note: This should only be called directly after the classifier has classified an image (so the blobs are valid)

        Arguments:
        layer -- the layer to visualize

        Keyword arguments:
        max_width -- the maximum width for vis_squares
        """
        if not self.loaded_model():
            return None, None

        activations = None
        weights = None

        normalize = True
        # don't normalize softmax layers
        if layer.type == 'Softmax':
            normalize = False

        if (not layer.bottom or layer.bottom[0] != layer.top[0]) and layer.top[0] in self.classifier.blobs:
            blob = self.classifier.blobs[layer.top[0]]
            assert blob.data.ndim == 4, 'expect blob.data.ndim == 4'
            if blob.data.shape[0] == 10:
                # 4 is the center crop (if oversampled)
                data = blob.data[4]
            else:
                data = blob.data[0]

            if data.shape[0] == 3:
                # can display as color channels
                # (1,height,width,channels)
                data = data.transpose(1,2,0)
                data = data[np.newaxis,...]

            # chop off data so that it will fit within max_width
            width = data.shape[2]
            if width > max_width:
                data = data[np.newaxis,0,:max_width,:max_width]
            else:
                if width > 1:
                    padsize = 1
                    width += 1
                else:
                    padsize = 0
                n = max_width/width
                n *= n
                data = data[:n]

            activations = utils.image.vis_square(data,
                    padsize     = padsize,
                    normalize   = normalize,
                    )
        if layer.name in self.classifier.params:
            params = self.classifier.params[layer.name][0]
            assert params.data.ndim == 4, 'expect params.ndim == 4'
            data = params.data
            if data.shape[1] == 3:
                # can display as color channels
                data = data.transpose(0,2,3,1)
            else:
                data = data.reshape((data.shape[0]*data.shape[1],data.shape[2],data.shape[3]))

            # chop off data so that it will fit within max_width
            width = data.shape[2]
            if width >= max_width:
                data = data[np.newaxis,0,:max_width,:max_width]
            else:
                if width > 1:
                    padsize = 1
                    width += 1
                else:
                    padsize = 0
                n = max_width/width
                n *= n
                data = data[:n]

            weights = utils.image.vis_square(data,
                    padsize     = padsize,
                    normalize   = normalize,
                    )
        return activations, weights

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
        images -- an array of np.arrays

        Keyword arguments:
        snapshot_epoch -- which snapshot to use
        """
        if not self.load_model(snapshot_epoch):
            return (None, None)
        if not self.read_labels():
            return (None, None)

        caffe_images = []
        for image in images:
            # Convert to float32 in [0,1] range
            image = image.astype('float32')/255.0
            if image.ndim == 2:
                image = image[:,:,np.newaxis]
            caffe_images.append(image)

        scores = self.classifier.predict(caffe_images, oversample=False)
        return (self.labels, scores)

    def has_model(self):
        """
        Returns True if there is a model that can be used
        """
        return len(self.snapshots) != 0

    def loaded_model(self):
        """
        Returns True if a model has been loaded
        """
        return self.loaded_snapshot_file is not None

    def load_model(self, epoch=None):
        """
        Loads a .caffemodel
        Returns True if the model is loaded (or if it was already loaded)

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

        assert file_to_load is not None

        if self.loaded_snapshot_file and self.loaded_snapshot_file == file_to_load:
            # Already loaded
            return True

        ### Do the load

        if self.image_mean is None:
            with open(self.dataset.path(self.dataset.train_db_task().mean_file), 'r') as f:
                blob = caffe_pb2.BlobProto()
                blob.MergeFromString(f.read())
                self.image_mean = np.reshape(blob.data, (
                        self.dataset.image_dims[2],
                        self.dataset.image_dims[0],
                        self.dataset.image_dims[1],
                        )
                        ).mean(1).mean(1) # 1 pixel

        if self.dataset.image_dims[2] == 3:
            channel_swap = (2,1,0)
        else:
            channel_swap = None

        self.classifier = caffe.Classifier(self.path(self.deploy_file), self.path(file_to_load),
                image_dims      = (
                    self.dataset.image_dims[0],
                    self.dataset.image_dims[1],
                    ),
                mean            = self.image_mean,
                raw_scale       = 255,
                channel_swap    = channel_swap,
                )

        self.loaded_snapshot_epoch = epoch
        self.loaded_snapshot_file = file_to_load
        return True

