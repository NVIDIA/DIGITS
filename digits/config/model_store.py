# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from . import config_option
from . import prompt

class ModelStoreOption(config_option.Option):

    def __init__(self):
        super(ModelStoreOption, self).__init__()
        self._config_dict_value = {
            'base_url': 'http://localhost',
            'port': '5050',
        }

    @staticmethod
    def config_file_key():
        return 'model_store'

    @classmethod
    def prompt_title(cls):
        return 'Model Store'

    @classmethod
    def visibility(cls):
        return config_option.Visibility.NEVER

    def optional(self):
        return True

    def suggestions(self):
        url = 'http://localhost:5050'
        return [prompt.Suggestion(url, 'H', desc='URL for Model Store', default=True)]

