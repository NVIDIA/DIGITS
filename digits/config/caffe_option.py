# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import sys
import imp
import platform
import subprocess

from digits import device_query
from digits.utils import parse_version
import config_option
import prompt

class CaffeOption(config_option.FrameworkOption):
    @staticmethod
    def config_file_key():
        return 'caffe_root'

    @classmethod
    def prompt_title(cls):
        return 'Caffe'

    @classmethod
    def prompt_message(cls):
        return 'Where is caffe installed?'

    def optional(self):
        #TODO: make this optional
        return False

    def suggestions(self):
        suggestions = []
        if 'CAFFE_ROOT' in os.environ:
            d = os.environ['CAFFE_ROOT']
            try:
                suggestions.append(prompt.Suggestion(
                    self.validate(d), 'R',
                    desc='CAFFE_ROOT', default=True))
            except config_option.BadValue:
                pass
        if 'CAFFE_HOME' in os.environ:
            d = os.environ['CAFFE_HOME']
            try:
                default = True
                if len(suggestions) > 0:
                    default = False
                suggestions.append(prompt.Suggestion(
                    self.validate(d), 'H',
                    desc='CAFFE_HOME', default=default))
            except config_option.BadValue:
                pass
        suggestions.append(prompt.Suggestion('<PATHS>', 'P',
            desc='PATH/PYTHONPATH', default=True))
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
            executable = cls.find_executable('caffe')
            if not executable:
                executable = cls.find_executable('caffe.exe')
            if not executable:
                raise config_option.BadValue('caffe binary not found in PATH')
            cls.validate_version(executable)

            # Find the python module
            try:
                imp.find_module('caffe')
            except ImportError:
                raise config_option.BadValue('caffe python package not found in PYTHONPATH')
            return value
        else:
            # Find the executable
            value = os.path.abspath(value)
            if not os.path.isdir(value):
                raise config_option.BadValue('"%s" is not a directory' % value)
            expected_path = os.path.join(value, 'build', 'tools', 'caffe')
            if not os.path.exists(expected_path):
                raise config_option.BadValue('caffe binary not found at "%s"' % value)
            cls.validate_version(expected_path)

            # Find the python module
            pythonpath = os.path.join(value, 'python')
            sys.path.insert(0, pythonpath)
            try:
                imp.find_module('caffe')
            except ImportError as e:
                raise config_option.BadValue('Error while importing caffe from "%s": %s' % (
                    pythonpath, e.message))
            finally:
                # Don't actually add this until apply() is called
                sys.path.pop(0)

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
        minimum_version = parse_version(0,11,0)
        version = cls.get_version(executable)

        if version is None:
            raise config_option.BadValue('Could not get version information from caffe at "%s". Are you using the NVIDIA fork?'
                    % executable)
        elif minimum_version > version:
            raise config_option.BadValue('Required version "%s" is greater than "%s". Upgrade your installation.'
                    % (
                        '.'.join(str(n) for n in minimum_version),
                        '.'.join(str(n) for n in version)
                        ))
        else:
            return True

    @staticmethod
    def get_version(executable):
        """
        Returns the caffe version as a (MAJOR, MINOR, PATCH) tuple or None

        Arguments:
        executable -- path to a caffe executable
        """
        # TODO: check `caffe --version` when it's implemented

        NVIDIA_SUFFIX = '-nv'

        if platform.system() == 'Linux':
            p = subprocess.Popen(['ldd', executable],
                    stdout = subprocess.PIPE,
                    stderr = subprocess.PIPE)
            if p.wait():
                raise config_option.BadValue(p.stderr.read().strip())
            else:
                libname = 'libcaffe'
                caffe_line = None

                # Search output for caffe library
                for line in p.stdout:
                    if libname in line:
                        caffe_line = line
                        break
                if caffe_line is None:
                    raise config_option.BadValue('%s not found in ldd output' % libname)

                # Read the symlink for libcaffe from ldd output
                symlink = caffe_line.split()[2]
                filename = os.path.basename(os.path.realpath(symlink))

                # Check for the nvidia suffix
                if NVIDIA_SUFFIX not in filename:
                    raise config_option.BadValue('Library at "%s" does not have expected suffix "%s". Are you using the NVIDIA/caffe fork?'
                            % (filename, NVIDIA_SUFFIX))

                # parse the version string
                match = re.match(r'%s%s\.so\.(\S+)$'
                        % (libname, NVIDIA_SUFFIX), filename)
                if match:
                    version_str = match.group(1)
                    return parse_version(version_str)
                else:
                    return None

        elif platform.system() == 'Darwin':
            # XXX: guess and let the user figure out errors later
            return parse_version(0,11,0)
        elif platform.system() == 'Windows':
            # XXX: guess and let the user figure out errors later
            return parse_version(0,11,0)
        else:
            print 'WARNING: platform "%s" not supported' % platform.system()
            return None

    def _set_config_dict_value(self, value):
        if not value:
            self._config_dict_value = None
        else:
            if value == '<PATHS>':
                executable = self.find_executable('caffe')
                if not executable:
                    executable = self.find_executable('caffe.exe')
            else:
                executable = os.path.join(value, 'build', 'tools', 'caffe')

            version = self.get_version(executable)

            if version >= parse_version(0,12):
                multi_gpu = True
            else:
                multi_gpu = False

            # TODO: ask caffe for this information
            cuda_enabled = len(device_query.get_devices()) > 0

            self._config_dict_value = {
                    'executable':   executable,
                    'version':      version,
                    'multi_gpu':    multi_gpu,
                    'cuda_enabled': cuda_enabled,
                    }

    def apply(self):
        if self._config_file_value:
            # Suppress GLOG output for python bindings
            GLOG_minloglevel = os.environ.pop('GLOG_minloglevel', None)
            # Show only "ERROR" and "FATAL"
            os.environ['GLOG_minloglevel'] = '2'

            if self._config_file_value != '<PATHS>':
                # Add caffe/python to PATH
                p = os.path.join(self._config_file_value, 'python')
                sys.path.insert(0, p)
                # Add caffe/python to PYTHONPATH
                #   so that build/tools/caffe is aware of python layers there
                os.environ['PYTHONPATH'] = '%s:%s' % (p, os.environ.get('PYTHONPATH'))

            try:
                import caffe
            except ImportError:
                print 'Did you forget to "make pycaffe"?'
                raise

            # Strange issue with protocol buffers and pickle - see issue #32
            sys.path.insert(0, os.path.join(
                os.path.dirname(caffe.__file__), 'proto'))

            # Turn GLOG output back on for subprocess calls
            if GLOG_minloglevel is None:
                del os.environ['GLOG_minloglevel']
            else:
                os.environ['GLOG_minloglevel'] = GLOG_minloglevel


