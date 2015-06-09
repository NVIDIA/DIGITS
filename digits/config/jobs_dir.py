# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import os
import tempfile

import digits
import config_option
import prompt

class JobsDirOption(config_option.Option):
    @staticmethod
    def config_file_key():
        return 'jobs_dir'

    @classmethod
    def prompt_title(cls):
        return 'Jobs Directory'

    @classmethod
    def prompt_message(cls):
        return 'Where would you like to store job data?'

    def suggestions(self):
        d = os.path.join(
                os.path.dirname(digits.__file__),
                'jobs')
        return [prompt.Suggestion(d, 'D', desc='default', default=True)]

    @staticmethod
    def is_path():
        return True

    @staticmethod
    def has_test_value():
        return True

    @staticmethod
    def test_value():
        return tempfile.mkdtemp()

    @classmethod
    def validate(cls, value):
        value = os.path.abspath(value)
        if os.path.exists(value):
            if not os.path.isdir(value):
                raise config_option.BadValue('Is not a directory')
            if not os.access(value, os.W_OK):
                raise config_option.BadValue('You do not have write permission')
            return value
        if not os.path.exists(os.path.dirname(value)):
            raise config_option.BadValue('Parent directory does not exist')
        if not os.access(os.path.dirname(value), os.W_OK):
            raise config_option.BadValue('You do not have write permission')
        return value

    def apply(self):
        if not os.path.exists(self._config_file_value):
            # make the directory
            os.mkdir(self._config_file_value)
