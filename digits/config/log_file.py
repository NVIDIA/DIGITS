# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

from . import config_option
from . import prompt
import digits

class LogFileOption(config_option.Option):
    @staticmethod
    def config_file_key():
        return 'log_file'

    @classmethod
    def prompt_title(cls):
        return 'Log File'

    @classmethod
    def prompt_message(cls):
        return 'Where do you want the log files to be stored?'

    def optional(self):
        # if not set, no log will be saved
        return True

    def suggestions(self):
        suggested_dir = os.path.dirname(digits.__file__)

        if os.access(suggested_dir, os.W_OK):
            return [prompt.Suggestion(
                os.path.join(suggested_dir, 'digits.log'), 'D',
                desc='default', default=True)
                ]
        else:
            return []

    @staticmethod
    def is_path():
        return True

    @staticmethod
    def has_test_value():
        return True

    @staticmethod
    def test_value():
        return None

    @classmethod
    def validate(cls, value):
        if not value:
            return value
        value = os.path.abspath(value)
        dirname = os.path.dirname(value)

        if os.path.isfile(value):
            if not os.access(value, os.W_OK):
                raise config_option.BadValue('You do not have write permissions')
            if not os.access(dirname, os.W_OK):
                raise config_option.BadValue('You do not have write permissions for "%s"' % dirname)
            return value
        elif os.path.isdir(value):
            raise config_option.BadValue('"%s" is a directory' % value)
        else:
            if os.path.isdir(dirname):
                if not os.access(dirname, os.W_OK):
                    raise config_option.BadValue('You do not have write permissions for "%s"' % dirname)
                # filename is in a valid directory
                return value
            previous_dir = os.path.dirname(dirname)
            if not os.path.isdir(previous_dir):
                raise config_option.BadValue('"%s" not found' % value)
            if not os.access(previous_dir, os.W_OK):
                raise config_option.BadValue('You do not have write permissions for "%s"' % previous_dir)
            # the preceding directory can be created later (in apply())
            return value

    def apply(self):
        if not self._config_file_value:
            return

        dirname = os.path.dirname(self._config_file_value)
        if not os.path.exists(dirname):
            os.mkdir(dirname)


