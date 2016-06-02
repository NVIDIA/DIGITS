# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import imp
import os
import platform
import re
import subprocess
import sys

from . import config_option
from . import prompt
from digits import device_query
from digits.utils import parse_version
from digits.utils.errors import UnsupportedPlatformError

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
        nvidia_minimum_version = '0.11.0'
        info_dict = cls.get_info(executable)
        if info_dict['ver_str'] is None:
            raise config_option.BadValue('Your Caffe does not have version info.  Please upgrade it.')
        else:
            flavor = CaffeOption.get_flavor(info_dict['ver_str'])
            if flavor == 'NVIDIA' and parse_version(nvidia_minimum_version) > parse_version(info_dict['ver_str']):
                raise config_option.BadValue(
                    'Required version "{min_ver}" is greater than "{running_ver}".  '\
                    'Upgrade your installation.'\
                    .format(min_ver = nvidia_minimum_version, running_ver = info_dict['ver_str']))
            else:
                return True

    @staticmethod
    def get_executable_version_string(executable):
        """
        Returns the caffe version as either a string from results of command line option '-version'
        or None if '-version' not implemented

        Arguments:
        executable -- path to a caffe executable
        """

        supported_platforms = ['Windows', 'Linux', 'Darwin']
        version_string = None
        if platform.system() in supported_platforms:
            if platform.system() == 'Darwin':
                version_string = '0.11.0'
            else:
                p = subprocess.Popen([executable, '-version'],
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
                if p.wait():
                    raise config_option.BadValue(p.stderr.read().strip())
                else:
                    pattern = 'version'
                    for line in p.stdout:
                        if pattern in line:
                            version_string = line[line.find(pattern) + len(pattern)+1:].rstrip()
                            break
                    try:
                        parse_version(version_string)
                    except ValueError: #version_string is either ill-formatted or 'CAFFE_VERSION'
                        version_string = None
            return version_string
        else:
            raise UnsupportedPlatformError('platform "%s" not supported' % platform.system())

    @staticmethod
    def get_linked_library_version_string(executable):
        """
        Returns the information about executable's linked library name version
        or None if error

        Arguments:
        executable -- path to a caffe executable
        """

        version_string = None

        p = subprocess.Popen(['ldd', executable],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        if p.wait():
            raise config_option.BadValue(p.stderr.read().strip())
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

        # parse the version string
        match = re.match(r'%s(.*)\.so\.(\S+)$'
                         % (libname), filename)
        if match:
            version_string = match.group(2)
        return version_string

    @staticmethod
    def get_flavor(ver_str):
        """
        Returns the information about caffe library enhancement (NVIDIA or BVLC)

        Arguments:
        ver_str -- version string that can identify enhancement flavor
        """
        if parse_version(0,99,0) > parse_version(ver_str) > parse_version(0,9,0):
            return 'NVIDIA'
        else:
            return 'BVLC'

    @staticmethod
    def get_version_string(executable):
        """
        Returns the caffe version as a string from executable or linked library name

        Arguments:
        executable -- path to a caffe executable
        """

        version_string = CaffeOption.get_executable_version_string(executable)
        if not version_string and platform.system() in ['Linux', 'Darwin']:
            version_string = CaffeOption.get_linked_library_version_string(executable)
        return version_string

    @staticmethod
    def get_info(executable):
        """
        Returns the caffe info a dict {'ver_str', 'flavor'}
            values of dict are None if unable to get version.

        Arguments:
        executable -- path to a caffe executable
        """
        try:
            version_string = CaffeOption.get_version_string(executable)
        except UnsupportedPlatformError:
            return {'ver_str': None, 'flavor': None}

        if version_string:
            return {'ver_str': version_string, 'flavor': CaffeOption.get_flavor(version_string)}
        else:
            return {'ver_str': None, 'flavor': None}

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

            info_dict = self.get_info(executable)
            version = parse_version(info_dict['ver_str'])
            if version >= parse_version(0,12):
                multi_gpu = True
            else:
                multi_gpu = False

            flavor = info_dict['flavor']
            # TODO: ask caffe for this information
            cuda_enabled = len(device_query.get_devices()) > 0

            self._config_dict_value = {
                    'executable':   executable,
                    'version':      version,
                    'ver_str':      info_dict['ver_str'],
                    'multi_gpu':    multi_gpu,
                    'cuda_enabled': cuda_enabled,
                    'flavor':       flavor
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

            # for Windows environment, loading h5py before caffe solves the issue mentioned in
            # https://github.com/NVIDIA/DIGITS/issues/47#issuecomment-206292824
            import h5py
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


