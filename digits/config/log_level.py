# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import config_option
import prompt

class LogLevelOption(config_option.Option):
    @staticmethod
    def config_file_key():
        return 'log_level'

    @classmethod
    def prompt_title(cls):
        return 'Log Level'

    @classmethod
    def prompt_message(cls):
        return 'What is the minimum log level that you want to save to your logfile? [error/warning/info/debug]'

    @classmethod
    def visibility(self):
        return config_option.Visibility.HIDDEN

    def suggestions(self):
        return [
                prompt.Suggestion('debug', 'D'),
                prompt.Suggestion('info', 'I', default=True),
                prompt.Suggestion('warning', 'W'),
                prompt.Suggestion('error', 'E'),
                ]

    @classmethod
    def validate(cls, value):
        value = value.strip().lower()
        if value not in ['error', 'warning', 'info', 'debug']:
            raise config_option.BadValue
        return value

