# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import re
import tempfile

from . import config_option
from . import prompt
import digits

class DatabaseUrlOption(config_option.Option):
    @staticmethod
    def config_file_key():
        return 'database_url'

    @classmethod
    def prompt_title(cls):
        return 'Database URL'

    @classmethod
    def prompt_message(cls):
        return 'How do you connect to your database?'

    def suggestions(self):
        sqlite_db = os.path.abspath(os.path.join(
            os.path.dirname(digits.__file__),
            'database', 'sqlite.db'))
        return [
            prompt.Suggestion('sqlite:///' + sqlite_db, 'S',
                              desc='SQLite', default=True),
            prompt.Suggestion('mysql:///digits_development', 'M', desc='MySQL'),
            prompt.Suggestion('postgresql:///digits_development', 'P', desc='PostgreSQL',
            ),
        ]

    @staticmethod
    def is_path():
        return True

    @staticmethod
    def has_test_value():
        return True

    @staticmethod
    def test_value():
        return 'sqlite:////tmp/digits_test.db'

    @classmethod
    def validate(cls, value):
        value = value.strip()
        # match "dialect[+driver]://*"
        match = re.match('[^\s\+]+(\+\S+)?:\/\/.*$', value)
        if not match:
            raise config_option.BadValue('The format is dialect+driver://username:password@host:port/database')

        return value
