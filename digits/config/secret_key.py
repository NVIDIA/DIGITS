# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import os

import config_option
import prompt

class SecretKeyOption(config_option.Option):
    @staticmethod
    def config_file_key():
        return 'secret_key'

    @classmethod
    def visibility(self):
        return config_option.Visibility.NEVER

    def suggestions(self):
        key = os.urandom(12).encode('hex')
        return [prompt.Suggestion(key, 'D', desc='default', default=True)]

