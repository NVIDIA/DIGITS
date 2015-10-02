# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import sys
import imp
import platform
import subprocess

from digits import device_query
import config_option
import prompt

class TorchOption(config_option.FrameworkOption):
    @staticmethod
    def config_file_key():
        return 'torch_root'

    @classmethod
    def prompt_title(cls):
        return 'Torch'

    @classmethod
    def prompt_message(cls):
        return 'Where is torch installed?'

    def optional(self):
        return True

    def suggestions(self):
        suggestions = []
        if 'TORCH_ROOT' in os.environ:
            d = os.environ['TORCH_ROOT']
            try:
                suggestions.append(prompt.Suggestion(
                    self.validate(d), 'R',
                    desc='TORCH_ROOT', default=True))
            except config_option.BadValue as e:
                print 'TORCH_ROOT "%s" is invalid:' % d
                print '\t%s' % e
        if 'TORCH_HOME' in os.environ:
            d = os.environ['TORCH_HOME']
            try:
                default = True
                if len(suggestions) > 0:
                    default = False
                suggestions.append(prompt.Suggestion(
                    self.validate(d), 'H',
                    desc='TORCH_HOME', default=default))
            except config_option.BadValue as e:
                print 'TORCH_HOME "%s" is invalid:' % d
                print '\t%s' % e
        suggestions.append(prompt.Suggestion('<PATHS>', 'P',
            desc='PATH/TORCHPATH', default=True))
        return suggestions

    @staticmethod
    def is_path():
        return True

    @classmethod
    def validate(cls, value):
        if not value:
            return value

        if value == '<PATHS>':
            # Find the executable
            executable = cls.find_executable('th')
            if not executable:
                raise config_option.BadValue('torch binary not found in PATH')
            #cls.validate_version(executable)
            return value
        else:
            # Find the executable
            value = os.path.abspath(value)
            if not os.path.isdir(value):
                raise config_option.BadValue('"%s" is not a directory' % value)
            expected_path = os.path.join(value, 'bin', 'th')
            if not os.path.exists(expected_path):
                raise config_option.BadValue('torch binary not found at "%s"' % value)
            #cls.validate_version(expected_path)
            return value

    @staticmethod
    def find_executable(program):
        """
        Finds an executable by searching through PATH
        Returns the path to the executable or None
        """
        for path in os.environ['PATH'].split(os.pathsep):
            path = path.strip('"')
            executable = os.path.join(path, program)
            if os.path.isfile(executable) and os.access(executable, os.X_OK):
                return executable
        return None

    @classmethod
    def validate_version(cls, executable):
        """
        Utility for checking the caffe version from within validate()
        Throws BadValue

        Arguments:
        executable -- path to a caffe executable
        """
        # Currently DIGITS don't have any restrictions on Torch version, so no need to implement this.
        pass

    def apply(self):
        pass
