# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import platform

from . import config_option
from . import prompt

class ServerNameOption(config_option.Option):
    @staticmethod
    def config_file_key():
        return 'server_name'

    @classmethod
    def prompt_title(cls):
        return 'Server Name'

    @classmethod
    def visibility(cls):
        return config_option.Visibility.HIDDEN

    def optional(self):
        return True

    def suggestions(self):
        hostname = platform.node()
        return [prompt.Suggestion(hostname, 'H', desc='HOSTNAME')]


