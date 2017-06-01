# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os.path
import re
import sys

import digits
from digits import utils
from digits.task import Task
from digits.utils import subclass, override

# NOTE: Increment this every time the pickled object
PICKLE_VERSION = 1


@subclass
class ParseFolderTask(Task):
    """Parses a folder into textfiles"""

    def __init__(self, folder, **kwargs):
        """
        Arguments:
        folder -- the folder to parse (can be a filesystem path or a url)

        Keyword arguments:
        percent_val -- percent of images used in the validation set
        percent_test -- percent of images used in the test set
        min_per_category -- minimum number of images per category
        max_per_category -- maximum number of images per category
        """
        # Take keyword arguments out of kwargs
        percent_val = kwargs.pop('percent_val', None)
        percent_test = kwargs.pop('percent_test', None)
        self.min_per_category = kwargs.pop('min_per_category', 2)
        self.max_per_category = kwargs.pop('max_per_category', None)

        super(ParseFolderTask, self).__init__(**kwargs)
        self.pickver_task_parsefolder = PICKLE_VERSION

        self.folder = folder

        if percent_val is None:
            self.percent_val = 0
        else:
            pct = float(percent_val)
            if pct < 0:
                pct = 0
            elif pct > 100:
                raise ValueError('percent_val must not exceed 100')
            self.percent_val = pct

        if percent_test is None:
            self.percent_test = 0
        else:
            pct = float(percent_test)
            if pct < 0:
                pct = 0
            elif pct > 100:
                raise ValueError('percent_test must not exceed 100')
            self.percent_test = pct

        if percent_val is not None and percent_test is not None and percent_val + percent_test > 100:
            raise ValueError('the sum of percent_val and percent_test must not exceed 100')

        self.train_file = utils.constants.TRAIN_FILE
        self.val_file = utils.constants.VAL_FILE
        self.labels_file = utils.constants.LABELS_FILE

        # Results

        self.train_count = None
        self.val_count = None
        self.test_count = None
        self.label_count = None

    def __getstate__(self):
        state = super(ParseFolderTask, self).__getstate__()
        return state

    def __setstate__(self, state):
        super(ParseFolderTask, self).__setstate__(state)

    @override
    def name(self):
        sets = []
        if (self.percent_val + self.percent_test) < 100:
            sets.append('train')
        if self.percent_val > 0:
            sets.append('val')
        if self.percent_test > 0:
            sets.append('test')

        return 'Parse Folder (%s)' % ('/'.join(sets))

    @override
    def html_id(self):
        sets = []
        if (self.percent_val + self.percent_test) < 100:
            sets.append('train')
        if self.percent_val > 0:
            sets.append('val')
        if self.percent_test > 0:
            sets.append('test')

        return 'task-parse-folder-%s' % ('-'.join(sets))

    @override
    def offer_resources(self, resources):
        key = 'parse_folder_task_pool'
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
            'tools', 'parse_folder.py'),
            self.folder,
            self.path(utils.constants.LABELS_FILE),
            '--min=%s' % self.min_per_category,
        ]

        if (self.percent_val + self.percent_test) < 100:
            args.append('--train_file=%s' % self.path(utils.constants.TRAIN_FILE))
        if self.percent_val > 0:
            args.append('--val_file=%s' % self.path(utils.constants.VAL_FILE))
            args.append('--percent_val=%s' % self.percent_val)
        if self.percent_test > 0:
            args.append('--test_file=%s' % self.path(utils.constants.TEST_FILE))
            args.append('--percent_test=%s' % self.percent_test)
        if self.max_per_category is not None:
            args.append('--max=%s' % self.max_per_category)

        return args

    @override
    def process_output(self, line):
        timestamp, level, message = self.preprocess_output_digits(line)
        if not message:
            return False

        # progress
        match = re.match(r'Progress: ([-+]?[0-9]*\.?[0-9]+(e[-+]?[0-9]+)?)', message)
        if match:
            self.progress = float(match.group(1))
            self.emit_progress_update()
            return True

        # totals
        match = re.match(r'Found (\d+) images in (\d+) categories', message)
        if match:
            self.label_count = int(match.group(2))
            return True

        # splits
        match = re.match(r'Selected (\d+) for (\w+)', message)
        if match:
            if match.group(2).startswith('training'):
                self.train_count = int(match.group(1))
            elif match.group(2).startswith('validation'):
                self.val_count = int(match.group(1))
            elif match.group(2).startswith('test'):
                self.test_count = int(match.group(1))
            return True

        if level == 'warning':
            self.logger.warning('%s: %s' % (self.name(), message))
            return True
        if level in ['error', 'critical']:
            self.logger.error('%s: %s' % (self.name(), message))
            self.exception = message
            return True

        return True
