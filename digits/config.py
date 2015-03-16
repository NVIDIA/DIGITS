#!/usr/bin/python
# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import sys
import math
import argparse
import imp
import subprocess
import tempfile

import ConfigParser

_caffe_exe = None #XXX hack for passing to GpusOption

### ConfigOption classes

class ConfigOption(object):
    """
    Base class for configuration options in this file
    """
    def __init__(self, level):
        self.level = level
        self._val = None

    @staticmethod
    def name():
        """
        Key in the config file for this option
        """
        raise NotImplementedError()

    @property
    def value(self):
        return self._val

    @value.setter
    def value(self, value):
        self._val = self.validate(value)

    @staticmethod
    def is_silent():
        """
        If True, set the value to the default without user input
        """
        return False

    def prompt_message(self):
        return 'Value for %s' % self.name()

    def default_value(self):
        return None

    def validate(self, value):
        """
        Returns the validated value
        Raises ValueErrors if invalid
        """
        return value

class CaffeRootOption(ConfigOption):
    @staticmethod
    def name():
        return 'caffe_root'

    def prompt_message(self):
        return 'Where is caffe installed? (enter "SYS" if installed system-wide)'

    def default_value(self):
        if 'CAFFE_HOME' in os.environ:
            #d = os.path.join(os.environ['CAFFE_HOME'], 'distribute')
            d = os.environ['CAFFE_HOME']
            try:
                return self.validate(d)
            except ValueError as e:
                print 'Guessed "%s" from CAFFE_HOME' % d
                print 'ERROR: %s' % e
        if 'CAFFE_ROOT' in os.environ:
            #d = os.path.join(os.environ['CAFFE_ROOT'], 'distribute')
            d = os.environ['CAFFE_ROOT']
            try:
                return self.validate(d)
            except ValueError as e:
                print 'Guessed "%s" from CAFFE_ROOT' % d
                print 'ERROR: %s' % e
        return 'SYS'

    def validate(self, value):
        global _caffe_exe
        if value == 'SYS':
            if not self.find_executable('caffe'):
                raise ValueError('caffe binary cannot be found')
            else:
                _caffe_exe = 'caffe'
            try:
                imp.find_module('caffe')
            except ImportError:
                raise ValueError('caffe python package cannot be found')
            return value
        else:
            value = os.path.normpath(value)
            if not os.path.isabs(value):
                raise ValueError('Must be an absolute path')
            if not os.path.exists(value):
                raise ValueError('Directory does not exist')
            if not os.path.isdir(value):
                raise ValueError('Must be a directory')
#            if not os.path.exists(os.path.join(value, 'bin', 'caffe.bin')):
            if not os.path.exists(os.path.join(value, 'build', 'tools', 'caffe.bin')):
                raise ValueError('Does not contain the caffe binary')

            _caffe_exe = os.path.join(value, 'build', 'tools', 'caffe.bin')

            #XXX remove other caffe/python paths from PATH
            sys.path = [os.path.join(value, 'python')] + [p for p in sys.path if os.path.join('caffe', 'python') not in p]
            try:
                imp.find_module('caffe')
            except ImportError:
                sys.path.pop(0)
                raise ValueError('caffe python package cannot be found')
            return value

    @staticmethod
    def find_executable(program):
        for path in os.environ['PATH'].split(os.pathsep):
            path = path.strip('"')
            executable = os.path.join(path, program)
            if os.path.isfile(executable) and os.access(executable, os.X_OK):
                return True
        return False

class GpusOption(ConfigOption):
    @staticmethod
    def name():
        return 'gpu_list'

    def prompt_message(self):
        s = 'Attached devices:\n'
        for gpu in self.query_gpus():
            s += 'Device #%s:\n' % gpu['device_number']
            s += '\tName: %s\n' % gpu['name']
            s += '\tCompute capability: %s.%s\n' % (gpu['major_revision'], gpu['minor_revision'])
            s += '\tMemory: %s\n' % self.convert_size(gpu['total_memory'])
            s += '\tMultiprocessors: %s\n' % gpu['mp_count']
            s += '\n'
        return s + '\nInput the IDs of the devices you would like to use, separated by commas, in order of preference.'

    def default_value(self):
        return ','.join([str(gpu['device_number']) for gpu in self.query_gpus()])

    def is_silent(self):
        try:
            return len(self.query_gpus()) == 0
        except ValueError:
            return False

    def validate(self, value):
        choices = []
        gpus = self.query_gpus()

        if not gpus:
            return ''
        if len(gpus) and not value.strip():
            raise ValueError('Empty list')
        for word in value.split(','):
            num = int(word)
            found = False
            for gpu in gpus:
                if gpu['device_number'] == num:
                    if 'chosen' in gpu:
                        raise ValueError('You cannot select a GPU twice')
                    else:
                        found = True
                        gpu['chosen'] = True
                        break
            if not found:
                raise ValueError('There is no GPU #%d' % num)
            else:
                choices.append(num)

        if choices > 0:
            return ','.join([str(num) for num in choices])
        else:
            raise ValueError('Empty list')

    @classmethod
    def convert_size(cls, size):
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size,1024)))
        p = math.pow(1024,i)
        s = round(size/p,2)
        if (s > 0):
            return '%s %s' % (s,size_name[i])
        else:
            return '0B'

    @classmethod
    def query_gpus(cls):
        """
        Uses caffe's device_query method to get information
        Returns an array of dicts

        If CUDA_VISIBLE_DEVICES if set, it will affect this list
        """
        gpus = []
        gpu_index = 0
        valid = True
        while valid:
            try:
                global _caffe_exe
                if _caffe_exe is None:
                    raise ValueError('Cannot query GPUs without a valid caffe_root')
                output = subprocess.check_output(
                        [_caffe_exe, 'device_query',
                            '-gpu', str(gpu_index)
                            ],
                        stderr = subprocess.STDOUT)
                gpu = {'device_number': gpu_index}
                for line in output.split('\n'):
                    match = re.match(r'.+\] (.+)$', line)
                    if match:
                        message = match.group(1)
                        if message.startswith('Major revision number'):
                            match = re.match(r'.+:\s+(\d+)\s*$', message)
                            gpu['major_revision'] = int(match.group(1))
                        elif message.startswith('Minor revision number'):
                            match = re.match(r'.+:\s+(\d+)\s*$', message)
                            gpu['minor_revision'] = int(match.group(1))
                        elif message.startswith('Name'):
                            match = re.match(r'.+:\s+(.+)*$', message)
                            gpu['name'] = match.group(1)
                        elif message.startswith('Total global memory'):
                            match = re.match(r'.+:\s+(\d+)\s*$', message)
                            gpu['total_memory'] = int(match.group(1))
                        elif message.startswith('Number of multiprocessors'):
                            match = re.match(r'.+:\s+(\d+)\s*$', message)
                            gpu['mp_count'] = int(match.group(1))
                gpus.append(gpu)
                gpu_index += 1
            except subprocess.CalledProcessError:
                valid = False
        return gpus

class JobsDirOption(ConfigOption):
    @staticmethod
    def name():
        return 'jobs_dir'

    def prompt_message(self):
        return 'Where would you like to store jobs?'

    def default_value(self):
        if self.level == 'system':
            return '/var/digits-jobs'
        elif self.level == 'user':
            return os.path.join(DigitsConfig.get_user_level_dir(), 'jobs')
        elif self.level == 'test':
            return tempfile.mkdtemp()
        else:
            raise ValueError('invalid level')

    def validate(self, value):
        value = os.path.normpath(value)
        if not os.path.isabs(value):
            raise ValueError('Must be an absolute path')
        if os.path.exists(value):
            if not os.path.isdir(value):
                raise ValueError('Is not a directory')
            if not os.access(value, os.W_OK):
                raise ValueError('You do not have write permissions')
            return value
        if not os.path.exists(os.path.dirname(value)):
            raise ValueError('Parent directory does not exist')
        if not os.access(os.path.dirname(value), os.W_OK):
            raise ValueError('You do not have write permissions')
        if not os.path.exists(value):
            # make the directory
            os.mkdir(value)
        return value

class LogLevelOption(ConfigOption):
    @staticmethod
    def name():
        return 'log_level'

    def prompt_message(self):
        return 'What is the minimum log level that you want to save to your logfile? [error/warning/info/debug]'

    def default_value(self):
        return 'info'

    def validate(self, value):
        value = value.strip().lower()
        if value not in ['error', 'warning', 'info', 'debug']:
            raise ValueError()
        return value

class SecretKeyOption(ConfigOption):
    @staticmethod
    def name():
        return 'secret_key'

    @staticmethod
    def is_silent():
        return True

    def default_value(self):
        return os.urandom(12).encode('hex')

### main class

class DigitsConfig:
    config_section = 'DiGiTS'

    def __init__(self, level):
        config_file_name = 'digits.cfg'
        log_file_name = 'digits.log'

        if level not in ['system', 'user', 'test']:
            raise ValueError('level must be "system" or "user" or "test"')
        self.level = level

        self.config_file = self.get_config_file(level)
        self.log_file = self.get_log_file(level)

        self.option_list = [
                CaffeRootOption(level),
                GpusOption(level),
                JobsDirOption(level),
                LogLevelOption(level),
                SecretKeyOption(level),
                ]

        self.options = None
        self.valid = False

    def system_level(self):
        return self.level == 'system'

    def user_level(self):
        return self.level == 'user'

    def test_level(self):
        return self.level == 'test'

    @classmethod
    def get_user_level_dir(cls):
        """
        If level == user, store things in this directory
        """
        d = os.path.join(os.environ['HOME'], '.digits')
        if not os.path.exists(d):
            os.mkdir(d)
        return d

    @classmethod
    def get_config_file(cls, level):
        filename = 'digits.cfg'
        if level == 'system':
            return os.path.join('/etc', filename)
        elif level == 'user':
            return os.path.join(cls.get_user_level_dir(), filename)
        elif level == 'test':
            _handle, _tempfilename = tempfile.mkstemp(suffix='.cfg')
            return _tempfilename
        else:
            raise ValueError('invalid level')

    @classmethod
    def get_log_file(cls, level):
        filename = 'digits.log'
        if level == 'system':
            return os.path.join('/var/log', filename)
        elif level == 'user':
            return os.path.join(cls.get_user_level_dir(), filename)
        elif level == 'test':
            _handle, _tempfilename = tempfile.mkstemp(suffix='.log')
            return _tempfilename
        else:
            raise ValueError('invalid level')

    def read_from_file(self):
        """
        Returns the current config in a dict or None
        """
        if not os.path.exists(self.config_file):
            return None
        cfg = ConfigParser.SafeConfigParser()
        try:
            cfg.read(self.config_file)
        except Exception as e:
            print '%s: %s' % (type(e).__name__, e)
            return None

        if not cfg.has_section(self.config_section):
            return None

        options = {}
        for key, val in cfg.items(self.config_section):
            options[key] = val
        return options

    def save_to_file(self):
        """
        Save options to file
        """
        cfg = ConfigParser.SafeConfigParser()
        cfg.add_section(self.config_section)
        for key, val in self.options.iteritems():
            cfg.set(self.config_section, key, val)
        with open(self.config_file, 'w') as outfile:
            cfg.write(outfile)

    def load(self):
        """
        Reads values from the config file
        Returns True if a valid config has been loaded
        """
        # validate permissions for config_file and log_file
        allowed = True
        if os.path.exists(self.config_file):
            if not os.access(self.config_file, os.R_OK):
                allowed = False
                print 'cannot write to %s' % self.config_file
        else:
            parent_dir = os.path.dirname(self.config_file)
            if not os.path.exists(parent_dir):
                allowed = False
                print '%s does not exist' % parent_dir
            if not os.access(parent_dir, os.W_OK):
                allowed = False
                print 'cannot write to %s' % parent_dir
        if os.path.exists(self.log_file):
            if not os.access(self.log_file, os.W_OK):
                allowed = False
                print 'cannot write to %s' % self.log_file
        else:
            parent_dir = os.path.dirname(self.log_file)
            if not os.path.exists(parent_dir):
                allowed = False
                print '%s does not exist' % parent_dir
            if not os.access(parent_dir, os.W_OK):
                allowed = False
                print 'cannot write to %s' % parent_dir
        if not allowed:
            print 'Did you mean to run this as root?'
            return False

        self.options = None
        self.valid = False

        old_config = self.read_from_file()
        if old_config is None:
            if self.user_level():
                return False
            else:
                old_config = {}

        valid = True
        dirty = False
        for option in self.option_list:
            if not option.name() in old_config:
                if option.is_silent():
                    option.value = option.default_value()
                    dirty = True
                else:
                    if self.user_level():
                        print 'config file missing option "%s"' % option.name()
                        valid = False
                    else:
                        try:
                            option.value = option.default_value()
                        except ValueError as e:
                            if self.test_level():
                                raise
                            print 'Cannot guess value for "%s": %s' % (option.name(), e)
                            valid = False

            else:
                value = old_config[option.name()]
                try:
                    option.value = value
                except ValueError as e:
                    if self.user_level():
                        print 'Invalid config option for "%s": %s' % (option.name(), e)
                        valid = False
                    else:
                        option.value = option.default_value()

        if not valid:
            return False

        self.options = {}
        for option in self.option_list:
            self.options[option.name()] = option.value

        if dirty:
            if not os.access(os.path.dirname(self.config_file), os.W_OK):
                print 'cannot write to %s' % self.config_file
                return False
            self.save_to_file()

        self.valid = True
        return True


    def prompt(self):
        """
        Prompts the user to fill out the configuration
        Returns True if a valid config was saved
        Returns False if the user cancels the process
        """
        self.options = None
        self.valid = False

        print 'Welcome to the DiGiTS config module.'
        print

        old_config = self.read_from_file()
        if old_config is None:
            old_config = {}

        for option in self.option_list:
            default = option.default_value()
            if option.name() in old_config:
                previous = old_config[option.name()]
            else:
                previous = None

            if option.is_silent():
                if previous is not None:
                    try:
                        option.value = previous
                    except ValueError:
                        option.value = default
                else:
                    option.value = default
                continue

            print option.prompt_message()

            previous_valid = False
            if option.name() in old_config:
                try:
                    option.value = previous
                    previous_valid = True
                except ValueError:
                    pass

            accept_previous = False
            if previous_valid:
                valid = False
                while not valid:
                    try:
                        print 'Accept previous value? [%s]' % previous
                        value = raw_input('(y/n/q) >>> ')
                        value = value.strip().lower()
                        if value.startswith('q'):
                            return False
                        if value.startswith('n'):
                            valid = True
                            accept_previous = False
                            print
                            print 'New value?'
                        elif value.startswith('y') or not value:
                            valid = True
                            accept_previous = True
                    except KeyboardInterrupt:
                        return False
            if accept_previous:
                print
                continue

            if default is not None:
                print '\t[default is %s]' % default

            valid = False
            while not valid:
                try:
                    value = raw_input('(q to quit) >>> ')
                except KeyboardInterrupt:
                    return False
                value = value.strip()
                if value.lower() in ['q', 'quit']:
                    return False
                if default is not None and not value:
                    value = default
                try:
                    option.value = value
                    valid = True
                except ValueError as e:
                    print 'ERROR:', e
                    print
            print

        self.options = {}
        for option in self.option_list:
            self.options[option.name()] = option.value

        print 'New config:'
        for option in self.options.iteritems():
            print '%20s - %s' % option
        print

        self.save_to_file()
        self.valid = True
        return True

    def clear(self):
        """
        Remove all user-set configuration options
        """
        self.options = None
        self.valid = False

        try:
            os.remove(self.config_file)
        except OSError:
            pass


current_config = None

def load_config():
    global current_config
    assert current_config is None, 'config already loaded'
    level = 'user'
    if 'DIGITS_LEVEL' in os.environ:
        level = os.environ['DIGITS_LEVEL']
    current_config = DigitsConfig(level)
    return current_config.load()

def valid_config():
    if current_config is None:
        return False
    else:
        return current_config.valid

def prompt_config():
    global current_config
    assert current_config is not None, 'config must be loaded first'
    return current_config.prompt()

def config_option(name):
    global current_config
    assert current_config is not None, 'config must be loaded first'
    assert current_config.valid, 'loaded config is invalid'
    if name == 'log_file':
        return current_config.log_file
    else:
        return current_config.options[name]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Config - DiGiTS')
    parser.add_argument('level',
            help='system/user/test')
    parser.add_argument('-c', '--clear',
            action='store_true',
            help='Clear the stored config before setting')
    args = vars(parser.parse_args())

    current_config = DigitsConfig(args['level'])

    if args['clear']:
        current_config.clear()
        print 'Config cleared.'

    if not current_config.prompt():
        sys.exit(1)

else:
    load_config()

