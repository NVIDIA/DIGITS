# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

from . import config_option
from . import prompt

class SecretKeyOption(config_option.Option):
    @staticmethod
    def config_file_key():
        return 'secret_key'

    @classmethod
    def visibility(cls):
        return config_option.Visibility.NEVER

    def suggestions(self):
        key = os.urandom(12).encode('hex')
        return [prompt.Suggestion(key, 'D', desc='default', default=True)]

