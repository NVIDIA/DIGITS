# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import operator
import os.path
import re
import sys

import digits
from digits import utils
from digits.task import Task
from digits.utils import subclass, override

# NOTE: Increment this every time the pickled version changes
PICKLE_VERSION = 3


@subclass
class CreateDbTask(Task):
    """Creates a database"""

    def __init__(self, input_file, db_name, backend, image_dims, **kwargs):
        """
        Arguments:
        input_file -- read images and labels from this file
        db_name -- save database to this location
        backend -- database backend (lmdb/hdf5)
        image_dims -- (height, width, channels)

        Keyword Arguments:
        image_folder -- prepend image paths with this folder
        shuffle -- shuffle images before saving
        resize_mode -- used in utils.image.resize_image()
        encoding -- 'none', 'png' or 'jpg'
        compression -- 'none' or 'gzip'
        mean_file -- save mean file to this location
        labels_file -- used to print category distribution
        """
        # Take keyword arguments out of kwargs
        self.image_folder = kwargs.pop('image_folder', None)
        self.shuffle = kwargs.pop('shuffle', True)
        self.resize_mode = kwargs.pop('resize_mode', None)
        self.encoding = kwargs.pop('encoding', None)
        self.compression = kwargs.pop('compression', None)
        self.mean_file = kwargs.pop('mean_file', None)
        self.labels_file = kwargs.pop('labels_file', None)

        super(CreateDbTask, self).__init__(**kwargs)
        self.pickver_task_createdb = PICKLE_VERSION

        self.input_file = input_file
        self.db_name = db_name
        self.backend = backend
        if backend == 'hdf5':
            # the list of hdf5 files is stored in a textfile
            self.textfile = os.path.join(self.db_name, 'list.txt')
        self.image_dims = image_dims
        if image_dims[2] == 3:
            self.image_channel_order = 'BGR'
        else:
            self.image_channel_order = None

        self.entries_count = None
        self.distribution = None
        self.create_db_log_file = "create_%s.log" % db_name

    def __getstate__(self):
        d = super(CreateDbTask, self).__getstate__()
        if 'create_db_log' in d:
            # don't save file handle
            del d['create_db_log']
        if 'labels' in d:
            del d['labels']
        return d

    def __setstate__(self, state):
        super(CreateDbTask, self).__setstate__(state)

        if self.pickver_task_createdb <= 1:
            if self.image_dims[2] == 1:
                self.image_channel_order = None
            elif self.encode:
                self.image_channel_order = 'BGR'
            else:
                self.image_channel_order = 'RGB'
        if self.pickver_task_createdb <= 2:
            if hasattr(self, 'encode'):
                if self.encode:
                    self.encoding = 'jpg'
                else:
                    self.encoding = 'none'
                delattr(self, 'encode')
            else:
                self.encoding = 'none'
        self.pickver_task_createdb = PICKLE_VERSION

        if not hasattr(self, 'backend') or self.backend is None:
            self.backend = 'lmdb'
        if not hasattr(self, 'compression') or self.compression is None:
            self.compression = 'none'

    @override
    def name(self):
        if self.db_name == utils.constants.TRAIN_DB or 'train' in self.db_name.lower():
            return 'Create DB (train)'
        elif self.db_name == utils.constants.VAL_DB or 'val' in self.db_name.lower():
            return 'Create DB (val)'
        elif self.db_name == utils.constants.TEST_DB or 'test' in self.db_name.lower():
            return 'Create DB (test)'
        else:
            return 'Create DB (%s)' % self.db_name

    @override
    def before_run(self):
        super(CreateDbTask, self).before_run()
        self.create_db_log = open(self.path(self.create_db_log_file), 'a')

    @override
    def html_id(self):
        if self.db_name == utils.constants.TRAIN_DB or 'train' in self.db_name.lower():
            return 'task-create_db-train'
        elif self.db_name == utils.constants.VAL_DB or 'val' in self.db_name.lower():
            return 'task-create_db-val'
        elif self.db_name == utils.constants.TEST_DB or 'test' in self.db_name.lower():
            return 'task-create_db-test'
        else:
            return super(CreateDbTask, self).html_id()

    @override
    def offer_resources(self, resources):
        key = 'create_db_task_pool'
        if key not in resources:
            return None
        for resource in resources[key]:
            if resource.remaining() >= 1:
                return {key: [(resource.identifier, 1)]}
        return None

    @override
    def task_arguments(self, resources, env):
        args = [sys.executable, os.path.join(
            os.path.dirname(os.path.abspath(digits.__file__)),
            'tools', 'create_db.py'),
            self.path(self.input_file),
            self.path(self.db_name),
            self.image_dims[1],
            self.image_dims[0],
            '--backend=%s' % self.backend,
            '--channels=%s' % self.image_dims[2],
            '--resize_mode=%s' % self.resize_mode,
        ]

        if self.mean_file is not None:
            args.append('--mean_file=%s' % self.path(self.mean_file))
            # Add a visual mean_file
            args.append('--mean_file=%s' % self.path(utils.constants.MEAN_FILE_IMAGE))
        if self.image_folder:
            args.append('--image_folder=%s' % self.image_folder)
        if self.shuffle:
            args.append('--shuffle')
        if self.encoding and self.encoding != 'none':
            args.append('--encoding=%s' % self.encoding)
        if self.compression and self.compression != 'none':
            args.append('--compression=%s' % self.compression)
        if self.backend == 'hdf5':
            args.append('--hdf5_dset_limit=%d' % 2**31)

        return args

    @override
    def process_output(self, line):
        from digits.webapp import socketio

        self.create_db_log.write('%s\n' % line)
        self.create_db_log.flush()

        timestamp, level, message = self.preprocess_output_digits(line)
        if not message:
            return False

        # progress
        match = re.match(r'Processed (\d+)\/(\d+)', message)
        if match:
            self.progress = float(match.group(1)) / int(match.group(2))
            self.emit_progress_update()
            return True

        # distribution
        match = re.match(r'Category (\d+) has (\d+)', message)
        if match and self.labels_file is not None:
            if not hasattr(self, 'distribution') or self.distribution is None:
                self.distribution = {}

            self.distribution[match.group(1)] = int(match.group(2))

            data = self.distribution_data()
            if data:
                socketio.emit('task update',
                              {
                                  'task': self.html_id(),
                                  'update': 'distribution',
                                  'data': data,
                              },
                              namespace='/jobs',
                              room=self.job_id,
                              )
            return True

        # result
        match = re.match(r'(\d+) images written to database', message)
        if match:
            self.entries_count = int(match.group(1))
            self.logger.debug(message)
            return True

        if level == 'warning':
            self.logger.warning('%s: %s' % (self.name(), message))
            return True
        if level in ['error', 'critical']:
            self.logger.error('%s: %s' % (self.name(), message))
            self.exception = message
            return True

        return True

    @override
    def after_run(self):
        from digits.webapp import socketio

        super(CreateDbTask, self).after_run()
        self.create_db_log.close()

        if self.backend == 'lmdb':
            socketio.emit('task update',
                          {
                              'task': self.html_id(),
                              'update': 'exploration-ready',
                          },
                          namespace='/jobs',
                          room=self.job_id,
                          )

        elif self.backend == 'hdf5':
            # add more path information to the list of h5 files
            lines = None
            with open(self.path(self.textfile)) as infile:
                lines = infile.readlines()
            with open(self.path(self.textfile), 'w') as outfile:
                for line in lines:
                    # XXX this works because the model job will be in an adjacent folder
                    outfile.write('%s\n' % os.path.join(
                        '..', self.job_id, self.db_name, line.strip()))

        if self.mean_file:
            socketio.emit('task update',
                          {
                              'task': self.html_id(),
                              'update': 'mean-image',
                              # XXX Can't use url_for here because we don't have a request context
                              'data': '/files/' + self.path('mean.jpg', relative=True),
                          },
                          namespace='/jobs',
                          room=self.job_id,
                          )

    def get_labels(self):
        """
        Read labels from labels_file and return them in a list
        """
        # The labels might be set already
        if hasattr(self, '_labels') and self._labels and len(self._labels) > 0:
            return self._labels

        assert hasattr(self, 'labels_file'), 'labels_file not set'
        assert self.labels_file, 'labels_file not set'
        assert os.path.exists(self.path(self.labels_file)), 'labels_file does not exist'

        labels = []
        with open(self.path(self.labels_file)) as infile:
            for line in infile:
                label = line.strip()
                if label:
                    labels.append(label)

        assert len(labels) > 0, 'no labels in labels_file'

        self._labels = labels
        return self._labels

    def distribution_data(self):
        """
        Returns distribution data for a C3.js graph
        """
        if self.distribution is None:
            return None
        try:
            labels = self.get_labels()
        except AssertionError:
            return None

        if len(self.distribution.keys()) != len(labels):
            return None

        values = ['Count']
        titles = []
        for key, value in sorted(
                self.distribution.items(),
                key=operator.itemgetter(1),
                reverse=True):
            values.append(value)
            titles.append(labels[int(key)])

        return {
            'data': {
                'columns': [values],
                'type': 'bar'
            },
            'axis': {
                'x': {
                    'type': 'category',
                    'categories': titles,
                }
            },
        }
