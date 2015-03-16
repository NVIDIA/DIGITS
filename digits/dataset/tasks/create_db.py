# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import sys
import os.path
import re
import operator

import digits
from digits import utils
from digits.utils import subclass, override
from digits.status import Status
from digits.task import Task

# NOTE: Increment this everytime the pickled version changes
PICKLE_VERSION = 1

@subclass
class CreateDbTask(Task):
    """Creates a database"""

    def __init__(self, input_file, db_name, image_dims, **kwargs):
        """
        Arguments:
        input_file -- read images and labels from this file
        db_name -- save database to this location
        image_dims -- (height, width, channels)

        Keyword Arguments:
        image_folder -- prepend image paths with this folder
        resize_mode -- used in utils.image.resize_image()
        encode -- save encoded JPEGs
        mean_file -- save mean file to this location
        backend -- type of database to use
        labels_file -- used to print category distribution
        """
        # Take keyword arguments out of kwargs
        self.image_folder = kwargs.pop('image_folder', None)
        self.resize_mode = kwargs.pop('resize_mode' , None)
        self.encode = kwargs.pop('encode', True)
        self.mean_file = kwargs.pop('mean_file', None)
        self.backend = kwargs.pop('backend', None)
        self.labels_file = kwargs.pop('labels_file', None)

        super(CreateDbTask, self).__init__(**kwargs)
        self.pickver_task_createdb = PICKLE_VERSION

        self.input_file = input_file
        self.db_name = db_name
        self.image_dims = image_dims

        self.entries_count = None
        self.distribution = None

    def __getstate__(self):
        d = super(CreateDbTask, self).__getstate__()
        if 'labels' in d:
            del d['labels']
        return d

    def __setstate__(self, state):
        super(CreateDbTask, self).__setstate__(state)

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
    def task_arguments(self, **kwargs):
        args = [sys.executable, os.path.join(os.path.dirname(os.path.dirname(digits.__file__)), 'tools', 'create_db.py'),
                self.path(self.input_file),
                self.path(self.db_name),
                self.image_dims[1],
                self.image_dims[0],
                '--channels=%s' % self.image_dims[2],
                '--resize_mode=%s' % self.resize_mode,
                ]

        if self.mean_file is not None:
            args.append('--mean_file=%s' % self.path(self.mean_file))
            # Add a visual mean_file
            args.append('--mean_file=%s' % self.path(utils.constants.MEAN_FILE_IMAGE))
        if self.image_folder:
            args.append('--image_folder=%s' % self.image_folder)
        if self.encode:
            args.append('--encode')

        return args

    @override
    def process_output(self, line):
        from digits.webapp import socketio

        timestamp, level, message = self.preprocess_output_digits(line)
        if not message:
            return False

        # progress
        match = re.match(r'Processed (\d+)\/(\d+)', message)
        if match:
            self.progress = float(match.group(1))/int(match.group(2))
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
        match = re.match(r'Total images added: (\d+)', message)
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

    def distribution_data(self):
        """
        Used to create data for a distribution graph
        Returns [[category, count], [category, count], ...]
        Returns None if distribution not present or incomplete
        """
        if self.distribution is None:
            return None
        if self.labels_file is None:
            return None
        if not hasattr(self, 'labels'):
            self.labels = []
            with open(self.path(self.labels_file), 'r') as infile:
                for line in infile:
                    line = line.strip()
                    if line:
                        self.labels.append(line)
        if len(self.distribution.keys()) != len(self.labels):
            return None

        data = [ ['Category', 'Count'] ]
        for key, value in sorted(
                self.distribution.items(),
                key=operator.itemgetter(1),
                reverse=True):
            data.append( [self.labels[int(key)], value] )
        return data

