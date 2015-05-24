#!/usr/bin/env python
# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import sys
import shutil
import math
import argparse
import imp
import subprocess
import tempfile
import platform
import readline
from collections import OrderedDict
import ConfigParser

import device_query

################################################################################
#   User input functions
################################################################################

def print_section_header(title):
    """
    Utility for printing a section header
    """
    print '{s:{c}^{n}}'.format(
            s = ' %s ' % title,
            # Extend to 80 characters
            n = 80,
            c = '=',
            )

def value_to_str(value):
    if value is None:
        return ''
    elif not value.strip():
        return '<NONE>'
    else:
        return value

class Suggestion(object):
    """
    A simple class for ConfigOption suggested values (used in get_input())
    """
    def __init__(self, value, char,
            desc = None,
            default = False,
            ):
        """
        Arguments:
        value -- the suggested value
        char -- a 1 character token representing this suggestion

        Keyword arguments:
        desc -- a short description of the source of this suggestion
        default -- if True, this is the suggestion that will be accepted by default
        """
        self.value = value
        if not isinstance(char, str):
            raise ValueError('char must be a string')
        if not (char == '' or len(char) == 1):
            raise ValueError('char must be a single character')
        self.char = char
        self.desc = desc
        self.default = default

    def __str__(self):
        s = '<Suggestion char="%s"' % self.char
        if self.desc:
            s += ' desc="%s"' % self.desc
        s += ' value="%s"' % self.value
        if self.default:
            s += ' default'
        s += '>'
        return s

def get_input(
        message     = None,
        validator   = None,
        suggestions = None,
        is_path     = False,
        ):
    """
    Gets input from the user
    Returns a valid value

    Keyword arguments:
    message -- printed first
    validator -- a function that returns True for a valid input
    suggestions -- a list of Suggestions
    is_path -- if True, tab autocomplete will be turned on
    """
    if suggestions is None:
        suggestions = []

    if message is not None:
        print message
    print

    # print list of suggestions
    max_width = 0
    for s in suggestions:
        if s.desc is not None and len(s.desc) > max_width:
            max_width = len(s.desc)
    if max_width > 0:
        print '\tSuggested values:'
        format_str = '\t%%-4s %%-%ds %%s' % (max_width+2,)
        default_found = False
        for s in suggestions:
            c = s.char
            if s.default and not default_found:
                default_found = True
                c += '*'
            desc = ''
            if s.desc is not None:
                desc = '[%s]' % s.desc
            print format_str % (('(%s)' % c), desc, value_to_str(s.value))

    if is_path:
        # turn on filename autocompletion
        delims = readline.get_completer_delims()
        readline.set_completer_delims(' \t\n;')
        readline.parse_and_bind('TAB: complete')

    user_input = None
    value = None
    valid = False
    while not valid:
        try:
            # Get user input
            user_input = raw_input('>> ').strip()
        except (KeyboardInterrupt, EOFError):
            print
            sys.exit(0)

        if user_input == '':
            for s in suggestions:
                if s.default:
                    print 'Using "%s"' % s.value
                    if s.value is not None and validator is not None:
                        try:
                            value = validator(s.value)
                            valid = True
                            break
                        except ValueError as e:
                            print 'ERROR:', e
                    else:
                        value = s.value
                        valid = True
                        break
        else:
            if len(user_input) == 1:
                for s in suggestions:
                    if s.char.lower() == user_input.lower():
                        print 'Using "%s"' % s.value
                        if s.value is not None and validator is not None:
                            try:
                                value = validator(s.value)
                                valid = True
                                break
                            except ValueError as e:
                                print 'ERROR:', e
                        else:
                            value = s.value
                            valid = True
                            break
            if not valid and validator is not None:
                if is_path:
                    user_input = os.path.expanduser(user_input)
                try:
                    value = validator(user_input)
                    valid = True
                    print 'Using "%s"' % value
                except ValueError as e:
                    print 'ERROR:', e

        if not valid:
            print 'Invalid input'

    if is_path:
        # back to normal
        readline.set_completer_delims(delims)
        readline.parse_and_bind('TAB: ')

    return value

################################################################################
#   ConfigOption classes
################################################################################

class ConfigOption(object):
    """
    Base class for configuration options
    """
    def __init__(self):
        self._val = None

    @staticmethod
    def name():
        """
        Key in the config file for this option
        """
        raise NotImplementedError()

    def prompt_message(self):
        """
        Printed before prompting
        """
        return None

    @classmethod
    def visibility(self):
        """
        -1  - Use default and never prompt user
        0   - Use default and only prompt user if -v is set
        1   - Prompt user unless -y is set
        2   - Prompt user even if -y is set
        """
        return 1

    def optional(self):
        """
        If True, then this option can be set to None
        """
        return False

    def suggestions(self):
        """
        Return a list of Suggestions
        """
        return []

    def is_path(self):
        """
        If True, tab autocompletion will be turned on during prompt
        """
        return False

    def default_value(self, suggestions=None):
        """
        Utility for retrieving the default value from the suggestion list
        """
        if suggestions is None:
            suggestions = self.suggestions()
        for s in suggestions:
            if s.default:
                return s.value
        return None

    def has_test_value(self):
        """
        If true, use test_value during testing
        """
        return False

    def test_value(self):
        """
        Returns a special value to be used during testing
        Ignores the current configuration
        """
        raise NotImplementedError

    @property
    def value(self):
        return self._val

    @value.setter
    def value(self, value):
        self._val = self.validate(value)

    @classmethod
    def validate(self, value):
        """
        Returns the validated value
        Raises ValueErrors if invalid
        """
        return value

    def apply(self):
        """
        Apply this configuration
        (may involve altering the PATH)
        """
        pass

class FrameworkOption(ConfigOption):
    """
    Base class for DL framework backends
    """
    def optional(self):
        return True

class CaffeRootOption(FrameworkOption):
    @staticmethod
    def name():
        return 'caffe_root'

    def prompt_message(self):
        return 'Where is caffe installed?'

    def optional(self):
        #TODO: make this optional
        return False

    def suggestions(self):
        suggestions = []
        if 'CAFFE_ROOT' in os.environ:
            d = os.environ['CAFFE_ROOT']
            try:
                self.validate(d)
                suggestions.append(Suggestion(d, 'R', desc='CAFFE_ROOT', default=True))
            except ValueError as e:
                print 'CAFFE_ROOT "%s" is invalid:' % d
                print '\t%s' % e
        if 'CAFFE_HOME' in os.environ:
            d = os.environ['CAFFE_HOME']
            try:
                self.validate(d)
                default = True
                if len(suggestions) > 0:
                    default = False
                suggestions.append(Suggestion(d, 'H', desc='CAFFE_HOME', default=default))
            except ValueError as e:
                print 'CAFFE_HOME "%s" is invalid:' % d
                print '\t%s' % e
        suggestions.append(Suggestion('<PATHS>', 'P', desc='PATH/PYTHONPATH', default=True))
        return suggestions

    def is_path(self):
        return True

    @classmethod
    def validate(cls, value):
        if not value:
            return value

        if value == '<PATHS>':
            caffe = cls.find_executable('caffe')
            if not caffe:
                raise ValueError('caffe binary not found in PATH')
            cls.validate_version(caffe)
            try:
                imp.find_module('caffe')
            except ImportError:
                raise ValueError('caffe python package not found in PYTHONPATH')
            return value
        else:
            value = os.path.abspath(value)
            if not os.path.isdir(value):
                raise ValueError('"%s" is not a directory' % value)
            #expected_path = os.path.join(value, 'bin', 'caffe.bin')
            expected_path = os.path.join(value, 'build', 'tools', 'caffe.bin')
            if not os.path.exists(expected_path):
                raise ValueError('caffe binary not found at "%s"' % value)
            cls.validate_version(expected_path)

            pythonpath = os.path.join(value, 'python')
            sys.path.insert(0, pythonpath)
            try:
                imp.find_module('caffe')
            except ImportError as e:
                raise ValueError('Error while importing caffe from "%s": %s' % (pythonpath, e.message))
            finally:
                # Don't actually add this until apply() is called
                sys.path.pop(0)

            return value

    # Used to validate the version
    REQUIRED_SUFFIX = '-nv'
    REQUIRED_VERSION = '0.11.0'

    @classmethod
    def validate_version(cls, executable):
        """
        Utility for checking the caffe version from within validate()

        Arguments:
        executable -- path to a caffe executable
        """
        # TODO: check `caffe --version` when it's implemented

        if platform.system() == 'Linux':
            p = subprocess.Popen(['ldd', executable],
                    stdout = subprocess.PIPE,
                    stderr = subprocess.PIPE)
            if p.wait():
                raise ValueError(p.stderr.read().strip())
            else:
                libname = 'libcaffe'
                for line in p.stdout:
                    if libname in line:
                        symlink = line.split()[2]
                        filename = os.path.basename(os.path.realpath(symlink))
                        if cls.REQUIRED_SUFFIX not in filename:
                            raise ValueError('Library at "%s" does not have expected suffix "%s". Are you building from the NVIDIA/caffe fork?' % (filename, cls.REQUIRED_SUFFIX))
                        if not filename.endswith(cls.REQUIRED_VERSION):
                            raise ValueError('Library at "%s" does not meet minimum library requirement "%s". Consider upgrading your NVIDIA/caffe installation.' % (filename, cls.REQUIRED_VERSION))
                        return True
            raise ValueError('%s not found in ldd output' % libname)
        elif platform.system() == 'Darwin':
            pass
        else:
            print 'WARNING: platform "%s" not supported' % platform.system()

    @staticmethod
    def find_executable(program):
        for path in os.environ['PATH'].split(os.pathsep):
            path = path.strip('"')
            executable = os.path.join(path, program)
            if os.path.isfile(executable) and os.access(executable, os.X_OK):
                return executable
        return None

    def apply(self):
        if self.value:
            # Suppress GLOG output for python bindings
            GLOG_minloglevel = os.environ.pop('GLOG_minloglevel', None)
            os.environ['GLOG_minloglevel'] = '5'

            if self.value != '<PATHS>':
                # Add caffe/python to PATH
                sys.path.insert(0, os.path.join(self.value, 'python'))
            try:
                import caffe
            except ImportError as e:
                print 'Did you forget to "make pycaffe"?'
                raise

            # Turn GLOG output back on for subprocess calls
            if GLOG_minloglevel is None:
                del os.environ['GLOG_minloglevel']
            else:
                os.environ['GLOG_minloglevel'] = GLOG_minloglevel


class GpuListOption(ConfigOption):
    @staticmethod
    def name():
        return 'gpu_list'

    def prompt_message(self):
        s = 'Attached devices:\n'
        for device_id, gpu in enumerate(device_query.get_devices()):
            s += 'Device #%s:\n' % device_id
            s += '\t%-20s %s\n' % ('Name', gpu.name)
            s += '\t%-20s %s.%s\n' % ('Compute capability', gpu.major, gpu.minor)
            s += '\t%-20s %s\n' % ('Memory', self.convert_size(gpu.totalGlobalMem))
            s += '\t%-20s %s\n' % ('Multiprocessors', gpu.multiProcessorCount)
            s += '\n'
        return s + '\nInput the IDs of the devices you would like to use, separated by commas, in order of preference.'

    def optional(self):
        return True

    def suggestions(self):
        if len(device_query.get_devices()) > 0:
            return [Suggestion(
                ','.join([str(x) for x in xrange(len(device_query.get_devices()))]),
                'D', desc='default', default=True)]
        else:
            return []

    @classmethod
    def visibility(self):
        if len(device_query.get_devices()) == 0:
            # Nothing to see here
            return -1
        if len(device_query.get_devices()) == 1:
            # Use just the one GPU by default
            return 0
        else:
            return 1

    @classmethod
    def validate(cls, value):
        if value == '':
            return value

        choices = []
        gpus = device_query.get_devices()

        if not gpus:
            return ''
        if len(gpus) and not value.strip():
            raise ValueError('Empty list')
        for word in value.split(','):
            if not word:
                continue
            num = int(word)
            found = False
            if not 0 <= num < len(gpus):
                raise ValueError('There is no GPU #%d' % num)
            if num in choices:
                raise ValueError('You cannot select a GPU twice')
            choices.append(num)

        if len(choices) > 0:
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

class JobsDirOption(ConfigOption):
    @staticmethod
    def name():
        return 'jobs_dir'

    def prompt_message(self):
        return 'Where would you like to store job data?'

    def suggestions(self):
        d = os.path.join(
                os.path.dirname(__file__),
                'jobs')
        return [Suggestion(d, 'D', desc='default', default=True)]

    def is_path(self):
        return True

    def has_test_value(self):
        return True

    def test_value(self):
        return tempfile.mkdtemp()

    @classmethod
    def validate(cls, value):
        value = os.path.abspath(value)
        if os.path.exists(value):
            if not os.path.isdir(value):
                raise ValueError('Is not a directory')
            if not os.access(value, os.W_OK):
                raise ValueError('You do not have write permission')
            return value
        if not os.path.exists(os.path.dirname(value)):
            raise ValueError('Parent directory does not exist')
        if not os.access(os.path.dirname(value), os.W_OK):
            raise ValueError('You do not have write permission')
        if not os.path.exists(value):
            # make the directory
            os.mkdir(value)
        return value

class LogFileOption(ConfigOption):
    @staticmethod
    def name():
        return 'log_file'

    def prompt_message(self):
        return 'Where do you want the log files to be stored?'

    def optional(self):
        # if not set, no log will be saved
        return True

    def suggestions(self):
        suggested_dir = os.path.dirname(__file__)

        if os.access(suggested_dir, os.W_OK):
            return [Suggestion(
                os.path.join(suggested_dir, 'digits.log'), 'D',
                desc='default', default=True)
                ]
        else:
            return []

    def is_path(self):
        return True

    def has_test_value(self):
        return True

    def test_value(self):
        return None

    @classmethod
    def validate(cls, value):
        if not value:
            return value
        value = os.path.abspath(value)
        dirname = os.path.dirname(value)

        if os.path.isfile(value):
            if not os.access(value, os.W_OK):
                raise ValueError('You do not have write permissions')
            if not os.access(dirname, os.W_OK):
                raise ValueError('You do not have write permissions for "%s"' % dirname)
            return value
        elif os.path.isdir(value):
            raise ValueError('"%s" is a directory' % value)
        else:
            if os.path.isdir(dirname):
                if not os.access(dirname, os.W_OK):
                    raise ValueError('You do not have write permissions for "%s"' % dirname)
                # filename is in a valid directory
                return value
            previous_dir = os.path.dirname(dirname)
            if not os.path.isdir(previous_dir):
                raise ValueError('"%s" not found' % value)
            if not os.access(previous_dir, os.W_OK):
                raise ValueError('You do not have write permissions for "%s"' % previous_dir)
            # the preceding directory can be created later (in apply())
            return value

    def apply(self):
        if not self.value:
            return

        dirname = os.path.dirname(self.value)
        if not os.path.exists(dirname):
            os.mkdir(dirname)


class LogLevelOption(ConfigOption):
    @staticmethod
    def name():
        return 'log_level'

    def prompt_message(self):
        return 'What is the minimum log level that you want to save to your logfile? [error/warning/info/debug]'

    @classmethod
    def visibility(self):
        return 0

    def suggestions(self):
        return [
                Suggestion('debug', 'D'),
                Suggestion('info', 'I', default=True),
                Suggestion('warning', 'W'),
                Suggestion('error', 'E'),
                ]

    @classmethod
    def validate(cls, value):
        value = value.strip().lower()
        if value not in ['error', 'warning', 'info', 'debug']:
            raise ValueError
        return value

class ServerNameOption(ConfigOption):
    @staticmethod
    def name():
        return 'server_name'

    @classmethod
    def visibility(self):
        return 0

    def optional(self):
        return True

    def suggestions(self):
        hostname = platform.node()
        return [Suggestion(hostname, 'H', desc='HOSTNAME')]


class SecretKeyOption(ConfigOption):
    @staticmethod
    def name():
        return 'secret_key'

    @classmethod
    def visibility(self):
        return -1

    def suggestions(self):
        key = os.urandom(12).encode('hex')
        return [Suggestion(key, 'D', desc='default', default=True)]

def optionClasses():
    """
    Returns a list of ConfigOption classes
    """
    return [
            JobsDirOption,
            GpuListOption,
            LogFileOption,
            LogLevelOption,
            ServerNameOption,
            SecretKeyOption,
            CaffeRootOption,
            ]

################################################################################
#   ConfigFile classes
################################################################################

class ConfigFile(object):
    """
    Handles IO on a config file
    """
    config_section = 'DIGITS'

    def __init__(self, filename):
        """
        Doesn't make a fuss if the file doesn't exist
        Use exists() to check
        """
        self._filename = filename
        self._options = OrderedDict()
        self.load()
        self._dirty = False

    def __str__(self):
        s = ''
        for item in self._options.iteritems():
            s += '%15s = %s\n' % item
        return s

    def filename(self):
        return self._filename

    def exists(self):
        """
        Returns True if the file exists
        """
        return os.path.isfile(self._filename)

    def can_read(self):
        """
        Returns True if the file can be read
        """
        return self.exists() and os.access(self._filename, os.R_OK)

    def can_write(self):
        """
        Returns True if the file can be written
        """
        if os.path.isfile(self._filename):
            return os.access(self._filename, os.W_OK)
        else:
            return os.access(
                    os.path.dirname(self._filename),
                    os.W_OK)

    def load(self):
        """
        Load options from the file
        Overwrites any values in self._options
        Returns True if the file loaded successfully
        """
        if not self.exists():
            return False
        cfg = ConfigParser.SafeConfigParser()
        cfg.read(self._filename)
        if not cfg.has_section(self.config_section):
            raise ValueError('expected section "%s" in config file at "%s"' % (
                self.config_section, self._filename))

        for key, val in cfg.items(self.config_section):
            self._options[key] = val
        return True

    def get(self, name):
        """
        Get a config option by name
        """
        if name in self._options:
            return self._options[name]
        else:
            return None

    def set(self, name, value):
        """
        Set a config option by name
        """
        if value is None:
            if name in self._options:
                del self._options[name]
                self._dirty = True
        else:
            if not (name in self._options and self._options[name] == value):
                self._dirty = True
            self._options[name] = value

    def dirty(self):
        """
        Returns True if there are changes to be written to disk
        """
        return self._dirty

    def save(self):
        """
        Save config file to disk
        """
        cfg = ConfigParser.SafeConfigParser()
        cfg.add_section(self.config_section)
        for name, value in self._options.iteritems():
            cfg.set(self.config_section, name, value)
        with open(self._filename, 'w') as outfile:
            cfg.write(outfile)


class SystemConfigFile(ConfigFile):
    def __init__(self):
        if platform.system() == 'Linux':
            filename = '/etc/digits.cfg'
        else:
            filename = None
        super(SystemConfigFile, self).__init__(filename)

class UserConfigFile(ConfigFile):
    def __init__(self, **kwargs):
        if 'HOME' in os.environ:
            filename = os.path.join(os.environ['HOME'], '.digits.cfg')
            old_filename = os.path.join(os.environ['HOME'], '.digits', 'digits.cfg')
            if not os.path.exists(filename) and os.path.exists(old_filename):
                try:
                    shutil.copyfile(old_filename, filename)
                    print 'Copied file at "%s" to "%s".' % (old_filename, filename)
                except (IOError, OSError):
                    pass
        else:
            filename = None
        super(UserConfigFile, self).__init__(filename)

class InstanceConfigFile(ConfigFile):
    def __init__(self):
        filename = os.path.join(os.path.dirname(__file__), 'digits.cfg')
        super(InstanceConfigFile, self).__init__(filename)

################################################################################
#   main functions
################################################################################

def print_config(verbose=False):
    """
    Prints out a matrix of config option values for each level
    """
    min_visibility = 1
    if verbose:
        min_visibility = 0
    columns = [[''] + [cls.name() for cls in optionClasses() if cls.visibility() >= min_visibility]]
    for header, cls in [
            ('INSTANCE', InstanceConfigFile),
            ('USER', UserConfigFile),
            ('SYSTEM', SystemConfigFile)]:
        cf = cls()
        if cf.can_read():
            column = [header]
            for key in columns[0][1:]:
                column.append(value_to_str(cf.get(key)))
            columns.append(column)

    if len(columns) == 1:
        # nothing to display
        return

    widths = []
    for column in columns:
        max_width = 0
        for item in column:
            width = len(str(item))
            if width > max_width:
                max_width = width
        widths.append(max_width)

    print_section_header('Current Config')

    for row_index in xrange(len(columns[0])):
        row = []
        for column_index in xrange(len(columns)):
            row.append(columns[column_index][row_index])
        format_str = ''
        for width in widths:
            format_str += '%%-%ds ' % width
        print format_str % tuple(row)
    print


def edit_config_file(verbose=False):
    """
    Prompt the user for which file to edit,
    then allow them to set options in that file
    """
    suggestions = []
    instanceConfig = InstanceConfigFile()
    if instanceConfig.can_write():
        suggestions.append(Suggestion(
            instanceConfig.filename(), 'I',
            desc = 'Instance', default=True))
    userConfig = UserConfigFile()
    if userConfig.can_write():
        suggestions.append(Suggestion(
            userConfig.filename(), 'U',
            desc = 'User', default=True))
    systemConfig = SystemConfigFile()
    if systemConfig.can_write():
        suggestions.append(Suggestion(
            systemConfig.filename(), 'S',
            desc = 'System', default=True))

    def filenameValidator(filename):
        """
        Returns True if this is a valid file to edit
        """
        if os.path.isfile(filename):
            if not os.access(filename, os.W_OK):
                raise ValueError('You do not have write permission')
            else:
                return filename

        if os.path.isdir(filename):
            raise ValueError('This is a directory')
        dirname = os.path.dirname(os.path.realpath(filename))
        if not os.path.isdir(dirname):
            raise ValueError('Path not found: %s' % dirname)
        elif not os.access(dirname, os.W_OK):
            raise ValueError('You do not have write permission')
        return filename

    filename = get_input(
            message     = 'Which file do you want to edit?',
            suggestions = suggestions,
            validator   = filenameValidator,
            is_path     = True,
            )

    print 'Editing file at %s ...' % os.path.realpath(filename)
    print

    is_standard_location = False

    if filename == instanceConfig.filename():
        is_standard_location = True
        instanceConfig = None
    if filename == userConfig.filename():
        is_standard_location = True
        userConfig = None
    if filename == systemConfig.filename():
        is_standard_location = True
        systemConfig = None

    configFile = ConfigFile(filename)

    for cls in optionClasses():
        option = cls()
        previous_value = configFile.get(option.name())
        suggestions = [Suggestion(None, 'U',
            desc='unset', default=(previous_value is None))]
        if previous_value is not None:
            suggestions.append(Suggestion(previous_value, '',
                desc = 'Previous', default = True))
        if instanceConfig is not None:
            instance_value = instanceConfig.get(option.name())
            if instance_value is not None:
                suggestions.append(Suggestion(instance_value, 'I',
                    desc = 'Instance', default = is_standard_location))
        if userConfig is not None:
            user_value = userConfig.get(option.name())
            if user_value is not None:
                suggestions.append(Suggestion(user_value, 'U',
                    desc = 'User', default = is_standard_location))
        if systemConfig is not None:
            system_value = systemConfig.get(option.name())
            if system_value is not None:
                suggestions.append(Suggestion(system_value, 'S',
                    desc = 'System', default = is_standard_location))
        suggestions += option.suggestions()
        if option.optional():
            suggestions.append(Suggestion('', 'N',
                desc = 'none', default = True))

        invisible = False
        if verbose and option.visibility() < 0:
            invisible = True
        elif not verbose and option.visibility() < 1:
            invisible = True

        if invisible:
            # Just don't set it
            pass
        else:
            print_section_header(option.name())
            value = get_input(
                    message     = option.prompt_message(),
                    validator   = option.validate,
                    suggestions = suggestions,
                    is_path     = option.is_path(),
                    )
            print
            configFile.set(option.name(), value)

    configFile.save()
    print 'New config saved at %s' % configFile.filename()
    print
    print configFile

current_config = None

def load_option(option, mode, newConfig,
        instanceConfig  =None,
        userConfig      = None,
        systemConfig    = None,
        ):
    """
    Called from load_config() [below]
    Returns the loaded value

    Arguments:
    option -- an instance of ConfigOption
    mode -- see docstring for load_config()
    newConfig -- an instance of ConfigFile
    instanceConfig -- the current InstanceConfigFile
    userConfig -- the current UserConfigFile
    systemConfig -- the current SystemConfigFile
    """
    if 'DIGITS_MODE_TEST' in os.environ and option.has_test_value():
        option.value = option.test_value()
        return option.value

    suggestions = []
    instance_value = instanceConfig.get(option.name())
    if instance_value is not None:
        suggestions.append(Suggestion(instance_value, '',
            desc = 'Previous', default = True))
    user_value = userConfig.get(option.name())
    if user_value is not None:
        suggestions.append(Suggestion(user_value, 'U',
            desc = 'User', default = True))
    system_value = systemConfig.get(option.name())
    if system_value is not None:
        suggestions.append(Suggestion(system_value, 'S',
            desc = 'System', default = True))
    suggestions += option.suggestions()
    if option.optional():
        suggestions.append(Suggestion('', 'N',
            desc = 'none', default = True))

    require_prompt = False
    if mode == 'verbose':
        if option.visibility() == -1:
            # check the default value
            default_value = option.default_value(suggestions)
            try:
                option.value = default_value
            except ValueError as e:
                print 'Default value for %s "%s" invalid:' % (option.name(), default_value)
                print '\t%s' % e
                require_prompt = True
        else:
            require_prompt = True

    if mode == 'quiet':
        # check the default value
        default_value = option.default_value(suggestions)
        try:
            option.value = default_value
        except ValueError as e:
            print 'Default value for %s "%s" invalid:' % (option.name(), default_value)
            print '\t%s' % e
            require_prompt = True

    if mode == 'force':
        # check for any valid default values
        valid = False
        for s in [s for s in suggestions if s.default]:
            try:
                option.value = s.value
                valid = True
                break
            except ValueError as e:
                print 'Default value for %s "%s" invalid:' % (option.name(), s.value)
                print '\t%s' % e
        if not valid:
            raise RuntimeError('No valid default value found for configuration option "%s"' % option.name())

    if require_prompt:
        print_section_header(option.name())
        value = get_input(
                message     = option.prompt_message(),
                validator   = option.validate,
                suggestions = suggestions,
                is_path     = option.is_path(),
                )
        print
        option.value = value
        newConfig.set(option.name(), option.value)

    return option.value

def load_config(mode='force'):
    """
    Load the current config
    By default, the user is prompted for values which have not been set already

    Keyword arguments:
    mode -- 3 options:
        verbose -- prompt for all options
        quiet -- prompt only for nonexistent or invalid options
        force -- throw errors for invalid options
    """
    global current_config
    current_config = {}

    instanceConfig = InstanceConfigFile()
    userConfig = UserConfigFile()
    systemConfig = SystemConfigFile()
    newConfig = InstanceConfigFile()

    non_framework_options = [cls() for cls in optionClasses()
            if not issubclass(cls, FrameworkOption)]
    framework_options = [cls() for cls in optionClasses()
            if issubclass(cls, FrameworkOption)]

    # Load non-framework config options
    for option in non_framework_options:
        load_option(option, mode, newConfig,
                instanceConfig, userConfig, systemConfig)

    has_one_framework = False
    verbose_for_frameworks = False
    while not has_one_framework:
        # Load framework config options
        if verbose_for_frameworks and mode == 'quiet':
            framework_mode = 'verbose'
        else:
            framework_mode = mode
        for option in framework_options:
            if load_option(option, framework_mode, newConfig,
                    instanceConfig, userConfig, systemConfig):
                has_one_framework = True

        if not has_one_framework:
            errstr = 'DIGITS requires at least one DL backend to run.'
            if mode == 'force':
                raise RuntimeError(errstr)
            else:
                print errstr
                # try again prompting all
                verbose_for_frameworks = True

    current_config = {}
    for option in non_framework_options + framework_options:
        option.apply()
        current_config[option.name()] = option.value

    if newConfig.dirty() and newConfig.can_write():
        newConfig.save()
        print 'Saved config to %s' % newConfig.filename()


def config_option(name):
    """
    Return the current configuration value for the given option

    Arguments:
    name -- the name of the configuration option
    """
    if current_config is None:
        raise RuntimeError('config must be loaded first')

    if name in current_config:
        return current_config[name]
    else:
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Config - DIGITS')
    parser.add_argument('-v', '--verbose',
            action="store_true",
            help='view more options')

    args = vars(parser.parse_args())

    print_config(args['verbose'])
    edit_config_file(args['verbose'])

elif 'DIGITS_MODE_TEST' in os.environ:
    load_config()

