# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os.path
import re
import sys

import digits
from digits.task import Task
from digits.utils import subclass, override

# NOTE: Increment this every time the pickled object
PICKLE_VERSION = 1


@subclass
class AnalyzeDbTask(Task):
    """
    Reads information from a database
    """

    def __init__(self, database, purpose, **kwargs):
        """
        Arguments:
        database -- path to the database to analyze
        purpose -- what is this database going to be used for

        Keyword arguments:
        force_same_shape -- if True, enforce that every entry in the database has the same shape
        """
        self.force_same_shape = kwargs.pop('force_same_shape', False)

        super(AnalyzeDbTask, self).__init__(**kwargs)
        self.pickver_task_analyzedb = PICKLE_VERSION

        self.database = database
        self.purpose = purpose
        self.backend = 'lmdb'

        # Results
        self.image_count = None
        self.image_width = None
        self.image_height = None
        self.image_channels = None

        self.analyze_db_log_file = 'analyze_db_%s.log' % '-'.join(p.lower() for p in self.purpose.split())

    def __getstate__(self):
        state = super(AnalyzeDbTask, self).__getstate__()
        if 'analyze_db_log' in state:
            del state['analyze_db_log']
        return state

    def __setstate__(self, state):
        super(AnalyzeDbTask, self).__setstate__(state)
        if not hasattr(self, 'backend') or self.backend is None:
            self.backend = 'lmdb'

    @override
    def name(self):
        return 'Analyze DB (%s)' % (self.purpose)

    @override
    def html_id(self):
        return 'task-analyze-db-%s' % '-'.join(p.lower() for p in self.purpose.split())

    @override
    def offer_resources(self, resources):
        key = 'analyze_db_task_pool'
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
            'tools', 'analyze_db.py'),
            self.database,
        ]
        if self.force_same_shape:
            args.append('--force-same-shape')
        else:
            args.append('--only-count')

        return args

    @override
    def before_run(self):
        super(AnalyzeDbTask, self).before_run()
        self.analyze_db_log = open(self.path(self.analyze_db_log_file), 'a')

    @override
    def process_output(self, line):
        self.analyze_db_log.write('%s\n' % line)
        self.analyze_db_log.flush()

        timestamp, level, message = self.preprocess_output_digits(line)
        if not message:
            return False

        # progress
        match = re.match(r'Progress: (\d+)\/(\d+)', message)
        if match:
            self.progress = float(match.group(1)) / float(match.group(2))
            self.emit_progress_update()
            return True

        # total count
        match = re.match(r'Total entries: (\d+)', message)
        if match:
            self.image_count = int(match.group(1))
            return True

        # image dimensions
        match = re.match(r'(\d+) entries found with shape ((\d+)x(\d+)x(\d+))', message)
        if match:
            # count = int(match.group(1))
            dims = match.group(2)
            self.image_width = int(match.group(3))
            self.image_height = int(match.group(4))
            self.image_channels = int(match.group(5))
            self.logger.debug('Images are %s' % dims)
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
        super(AnalyzeDbTask, self).after_run()
        self.analyze_db_log.close()

    def image_type(self):
        """
        Returns an easy-to-read version of self.image_channels
        """
        if self.image_channels is None:
            return None
        elif self.image_channels == 1:
            return 'GRAYSCALE'
        elif self.image_channels == 3:
            return 'COLOR'
        else:
            return '%s-channel' % self.image_channels
