# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import platform

import config_option
import prompt

class ServerNameOption(config_option.Option):
    @staticmethod
    def config_file_key():
        return 'server_name'

    @classmethod
    def prompt_title(cls):
        return 'Server Name'

    @classmethod
    def visibility(self):
        return config_option.Visibility.HIDDEN

    def optional(self):
        return True

    def suggestions(self):
        hostname = platform.node()
        return [prompt.Suggestion(hostname, 'H', desc='HOSTNAME')]


