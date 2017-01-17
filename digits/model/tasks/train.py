# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from collections import OrderedDict, namedtuple
import os.path
import time

import flask
import gevent
import psutil

from digits import device_query
from digits.task import Task
from digits.utils import subclass, override

# NOTE: Increment this every time the picked object changes
PICKLE_VERSION = 2

# Used to store network outputs
NetworkOutput = namedtuple('NetworkOutput', ['kind', 'data'])


@subclass
class TrainTask(Task):
    """
    Defines required methods for child classes
    """

    def __init__(self, job, dataset, train_epochs, snapshot_interval, learning_rate, lr_policy, **kwargs):
        """
        Arguments:
        job -- model job
        dataset -- a DatasetJob containing the dataset for this model
        train_epochs -- how many epochs of training data to train on
        snapshot_interval -- how many epochs between taking a snapshot
        learning_rate -- the base learning rate
        lr_policy -- a hash of options to be used for the learning rate policy

        Keyword arguments:
        gpu_count -- how many GPUs to use for training (integer)
        selected_gpus -- a list of GPU indexes to be used for training
        batch_size -- if set, override any network specific batch_size with this value
        batch_accumulation -- accumulate gradients over multiple batches
        val_interval -- how many epochs between validating the model with an epoch of validation data
        pretrained_model -- filename for a model to use for fine-tuning
        crop_size -- crop each image down to a square of this size
        use_mean -- subtract the dataset's mean file or mean pixel
        random_seed -- optional random seed
        data_aug -- data augmentation options
        """
        self.gpu_count = kwargs.pop('gpu_count', None)
        self.selected_gpus = kwargs.pop('selected_gpus', None)
        self.batch_size = kwargs.pop('batch_size', None)
        self.batch_accumulation = kwargs.pop('batch_accumulation', None)
        self.val_interval = kwargs.pop('val_interval', None)
        self.pretrained_model = kwargs.pop('pretrained_model', None)
        self.crop_size = kwargs.pop('crop_size', None)
        self.use_mean = kwargs.pop('use_mean', None)
        self.random_seed = kwargs.pop('random_seed', None)
        self.solver_type = kwargs.pop('solver_type', None)
        self.rms_decay = kwargs.pop('rms_decay', None)
        self.shuffle = kwargs.pop('shuffle', None)
        self.network = kwargs.pop('network', None)
        self.framework_id = kwargs.pop('framework_id', None)
        self.data_aug = kwargs.pop('data_aug', None)

        super(TrainTask, self).__init__(job_dir=job.dir(), **kwargs)
        self.pickver_task_train = PICKLE_VERSION

        self.job = job
        self.dataset = dataset
        self.train_epochs = train_epochs
        self.snapshot_interval = snapshot_interval
        self.learning_rate = learning_rate
        self.lr_policy = lr_policy

        self.current_epoch = 0
        self.snapshots = []

        # data gets stored as dicts of lists (for graphing)
        self.train_outputs = OrderedDict()
        self.val_outputs = OrderedDict()

    def __getstate__(self):
        state = super(TrainTask, self).__getstate__()
        if 'dataset' in state:
            del state['dataset']
        if 'snapshots' in state:
            del state['snapshots']
        if '_labels' in state:
            del state['_labels']
        if '_hw_socketio_thread' in state:
            del state['_hw_socketio_thread']
        return state

    def __setstate__(self, state):
        if state['pickver_task_train'] < 2:
            state['train_outputs'] = OrderedDict()
            state['val_outputs'] = OrderedDict()

            tl = state.pop('train_loss_updates', None)
            vl = state.pop('val_loss_updates', None)
            va = state.pop('val_accuracy_updates', None)
            lr = state.pop('lr_updates', None)
            if tl:
                state['train_outputs']['epoch'] = NetworkOutput('Epoch', [x[0] for x in tl])
                state['train_outputs']['loss'] = NetworkOutput('SoftmaxWithLoss', [x[1] for x in tl])
                state['train_outputs']['learning_rate'] = NetworkOutput('LearningRate', [x[1] for x in lr])
            if vl:
                state['val_outputs']['epoch'] = NetworkOutput('Epoch', [x[0] for x in vl])
                if va:
                    state['val_outputs']['accuracy'] = NetworkOutput('Accuracy', [x[1] / 100 for x in va])
                state['val_outputs']['loss'] = NetworkOutput('SoftmaxWithLoss', [x[1] for x in vl])

        if state['use_mean'] is True:
            state['use_mean'] = 'pixel'
        elif state['use_mean'] is False:
            state['use_mean'] = 'none'

        state['pickver_task_train'] = PICKLE_VERSION
        super(TrainTask, self).__setstate__(state)

        self.snapshots = []
        self.dataset = None

    @override
    def offer_resources(self, resources):
        if 'gpus' not in resources:
            return None
        if not resources['gpus']:
            return {}  # don't use a GPU at all
        if self.gpu_count is not None:
            identifiers = []
            for resource in resources['gpus']:
                if resource.remaining() >= 1:
                    identifiers.append(resource.identifier)
                    if len(identifiers) == self.gpu_count:
                        break
            if len(identifiers) == self.gpu_count:
                return {'gpus': [(i, 1) for i in identifiers]}
            else:
                return None
        elif self.selected_gpus is not None:
            all_available = True
            for i in self.selected_gpus:
                available = False
                for gpu in resources['gpus']:
                    if i == gpu.identifier:
                        if gpu.remaining() >= 1:
                            available = True
                        break
                if not available:
                    all_available = False
                    break
            if all_available:
                return {'gpus': [(i, 1) for i in self.selected_gpus]}
            else:
                return None
        return None

    @override
    def before_run(self):
        # start a thread which sends SocketIO updates about hardware utilization
        gpus = None
        if 'gpus' in self.current_resources:
            gpus = [identifier for (identifier, value) in self.current_resources['gpus']]

        self._hw_socketio_thread = gevent.spawn(
            self.hw_socketio_updater,
            gpus)

    def hw_socketio_updater(self, gpus):
        """
        This thread sends SocketIO messages about hardware utilization
        to connected clients

        Arguments:
        gpus -- a list of identifiers for the GPUs currently being used
        """
        from digits.webapp import app, socketio

        devices = []
        if gpus is not None:
            for index in gpus:
                device = device_query.get_device(index)
                if device:
                    devices.append((index, device))
                else:
                    raise RuntimeError('Failed to load gpu information for GPU #"%s"' % index)

        # this thread continues until killed in after_run()
        while True:
            # CPU (Non-GPU) Info
            data_cpu = {}
            if hasattr(self, 'p') and self.p is not None:
                data_cpu['pid'] = self.p.pid
                try:
                    ps = psutil.Process(self.p.pid)  # 'self.p' is the system call object
                    if ps.is_running():
                        if psutil.version_info[0] >= 2:
                            data_cpu['cpu_pct'] = ps.cpu_percent(interval=1)
                            data_cpu['mem_pct'] = ps.memory_percent()
                            data_cpu['mem_used'] = ps.memory_info().rss
                        else:
                            data_cpu['cpu_pct'] = ps.get_cpu_percent(interval=1)
                            data_cpu['mem_pct'] = ps.get_memory_percent()
                            data_cpu['mem_used'] = ps.get_memory_info().rss
                except psutil.NoSuchProcess:
                    # In rare case of instant process crash or PID went zombie (report nothing)
                    pass

            data_gpu = []
            for index, device in devices:
                update = {'name': device.name, 'index': index}
                nvml_info = device_query.get_nvml_info(index)
                if nvml_info is not None:
                    update.update(nvml_info)
                data_gpu.append(update)

            with app.app_context():
                html = flask.render_template('models/gpu_utilization.html',
                                             data_gpu=data_gpu,
                                             data_cpu=data_cpu)

                socketio.emit('task update',
                              {
                                  'task': self.html_id(),
                                  'update': 'gpu_utilization',
                                  'html': html,
                              },
                              namespace='/jobs',
                              room=self.job_id,
                              )
            gevent.sleep(1)

    def send_progress_update(self, epoch):
        """
        Sends socketio message about the current progress
        """
        if self.current_epoch == epoch:
            return

        self.current_epoch = epoch
        self.progress = epoch / self.train_epochs
        self.emit_progress_update()

    def save_train_output(self, *args):
        """
        Save output to self.train_outputs
        """
        from digits.webapp import socketio

        if not self.save_output(self.train_outputs, *args):
            return

        if self.last_train_update and (time.time() - self.last_train_update) < 5:
            return
        self.last_train_update = time.time()

        self.logger.debug('Training %s%% complete.' % round(100 * self.current_epoch / self.train_epochs, 2))

        # loss graph data
        data = self.combined_graph_data()
        if data:
            socketio.emit('task update',
                          {
                              'task': self.html_id(),
                              'update': 'combined_graph',
                              'data': data,
                          },
                          namespace='/jobs',
                          room=self.job_id,
                          )

            if data['columns']:
                # isolate the Loss column data for the sparkline
                graph_data = data['columns'][0][1:]
                socketio.emit('task update',
                              {
                                  'task': self.html_id(),
                                  'job_id': self.job_id,
                                  'update': 'combined_graph',
                                  'data': graph_data,
                              },
                              namespace='/jobs',
                              room='job_management',
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

    def save_val_output(self, *args):
        """
        Save output to self.val_outputs
        """
        from digits.webapp import socketio

        if not self.save_output(self.val_outputs, *args):
            return

        # loss graph data
        data = self.combined_graph_data()
        if data:
            socketio.emit('task update',
                          {
                              'task': self.html_id(),
                              'update': 'combined_graph',
                              'data': data,
                          },
                          namespace='/jobs',
                          room=self.job_id,
                          )

    def save_output(self, d, name, kind, value):
        """
        Save output to self.train_outputs or self.val_outputs
        Returns true if all outputs for this epoch have been added

        Arguments:
        d -- the dictionary where the output should be stored
        name -- name of the output (e.g. "accuracy")
        kind -- the type of outputs (e.g. "Accuracy")
        value -- value for this output (e.g. 0.95)
        """
        # don't let them be unicode
        name = str(name)
        kind = str(kind)

        # update d['epoch']
        if 'epoch' not in d:
            d['epoch'] = NetworkOutput('Epoch', [self.current_epoch])
        elif d['epoch'].data[-1] != self.current_epoch:
            d['epoch'].data.append(self.current_epoch)

        if name not in d:
            d[name] = NetworkOutput(kind, [])
        epoch_len = len(d['epoch'].data)
        name_len = len(d[name].data)

        # save to back of d[name]
        if name_len > epoch_len:
            raise Exception('Received a new output without being told the new epoch')
        elif name_len == epoch_len:
            # already exists
            if isinstance(d[name].data[-1], list):
                d[name].data[-1].append(value)
            else:
                d[name].data[-1] = [d[name].data[-1], value]
        elif name_len == epoch_len - 1:
            # expected case
            d[name].data.append(value)
        else:
            # we might have missed one
            for _ in xrange(epoch_len - name_len - 1):
                d[name].data.append(None)
            d[name].data.append(value)

        for key in d:
            if key not in ['epoch', 'learning_rate']:
                if len(d[key].data) != epoch_len:
                    return False
        return True

    @override
    def after_run(self):
        if hasattr(self, '_hw_socketio_thread'):
            self._hw_socketio_thread.kill()

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

    def get_snapshot(self, epoch=-1):
        """
        return snapshot file for specified epoch
        """
        snapshot_filename = None

        if len(self.snapshots) == 0:
            return "no snapshots"

        if epoch == -1 or not epoch:
            epoch = self.snapshots[-1][1]
            snapshot_filename = self.snapshots[-1][0]
        else:
            for f, e in self.snapshots:
                if e == epoch:
                    snapshot_filename = f
                    break
        if not snapshot_filename:
            raise ValueError('Invalid epoch')

        return snapshot_filename

    def get_snapshot_filename(self, epoch=-1):
        """
        Return the filename for the specified epoch
        """
        path, name = os.path.split(self.get_snapshot(epoch))
        return name

    def get_labels(self):
        """
        Read labels from labels_file and return them in a list
        """
        # The labels might be set already
        if hasattr(self, '_labels') and self._labels and len(self._labels) > 0:
            return self._labels

        assert hasattr(self.dataset, 'labels_file'), 'labels_file not set'
        assert self.dataset.labels_file, 'labels_file not set'
        assert os.path.exists(self.dataset.path(self.dataset.labels_file)), 'labels_file does not exist'

        labels = []
        with open(self.dataset.path(self.dataset.labels_file)) as infile:
            for line in infile:
                label = line.strip()
                if label:
                    labels.append(label)

        assert len(labels) > 0, 'no labels in labels_file'

        self._labels = labels
        return self._labels

    def lr_graph_data(self):
        """
        Returns learning rate data formatted for a C3.js graph

        Keyword arguments:

        """
        if not self.train_outputs or 'epoch' not in self.train_outputs or 'learning_rate' not in self.train_outputs:
            return None

        # return 100-200 values or fewer
        stride = max(len(self.train_outputs['epoch'].data) / 100, 1)
        e = ['epoch'] + self.train_outputs['epoch'].data[::stride]
        lr = ['lr'] + self.train_outputs['learning_rate'].data[::stride]

        return {
            'columns': [e, lr],
            'xs': {
                'lr': 'epoch'
            },
            'names': {
                'lr': 'Learning Rate'
            },
        }

    def combined_graph_data(self, cull=True):
        """
        Returns all train/val outputs in data for one C3.js graph

        Keyword arguments:
        cull -- if True, cut down the number of data points returned to a reasonable size
        """
        data = {
            'columns': [],
            'xs': {},
            'axes': {},
            'names': {},
        }

        added_train_data = False
        added_val_data = False

        if self.train_outputs and 'epoch' in self.train_outputs:
            if cull:
                # max 200 data points
                stride = max(len(self.train_outputs['epoch'].data) / 100, 1)
            else:
                # return all data
                stride = 1
            for name, output in self.train_outputs.iteritems():
                if name not in ['epoch', 'learning_rate']:
                    col_id = '%s-train' % name
                    data['xs'][col_id] = 'train_epochs'
                    data['names'][col_id] = '%s (train)' % name
                    if 'accuracy' in output.kind.lower() or 'accuracy' in name.lower():
                        data['columns'].append([col_id] + [
                            (100 * x if x is not None else 'none')
                            for x in output.data[::stride]])
                        data['axes'][col_id] = 'y2'
                    else:
                        data['columns'].append([col_id] + [
                            (x if x is not None else 'none')
                            for x in output.data[::stride]])
                    added_train_data = True
        if added_train_data:
            data['columns'].append(['train_epochs'] + self.train_outputs['epoch'].data[::stride])

        if self.val_outputs and 'epoch' in self.val_outputs:
            if cull:
                # max 200 data points
                stride = max(len(self.val_outputs['epoch'].data) / 100, 1)
            else:
                # return all data
                stride = 1
            for name, output in self.val_outputs.iteritems():
                if name not in ['epoch']:
                    col_id = '%s-val' % name
                    data['xs'][col_id] = 'val_epochs'
                    data['names'][col_id] = '%s (val)' % name
                    if 'accuracy' in output.kind.lower() or 'accuracy' in name.lower():
                        data['columns'].append([col_id] + [
                            (100 * x if x is not None else 'none')
                            for x in output.data[::stride]])
                        data['axes'][col_id] = 'y2'
                    else:
                        data['columns'].append([col_id] + [
                            (x if x is not None else 'none')
                            for x in output.data[::stride]])
                    added_val_data = True
        if added_val_data:
            data['columns'].append(['val_epochs'] + self.val_outputs['epoch'].data[::stride])

        if added_train_data:
            return data
        else:
            # return None if only validation data exists
            # helps with ordering of columns in graph
            return None

    # return id of framework used for training
    def get_framework_id(self):
        """
        Returns a string
        """
        return self.framework_id

    def get_model_files(self):
        """
        return path to model file
        """
        raise NotImplementedError()

    def get_network_desc(self):
        """
        return text description of model
        """
        raise NotImplementedError()

    def get_task_stats(self, epoch=-1):
        """
        return a dictionary of task statistics
        """
        raise NotImplementedError()
