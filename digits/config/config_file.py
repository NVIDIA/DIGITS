# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import os
import platform
import ConfigParser
from collections import OrderedDict

import digits

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
        return self._filename is not None and os.path.isfile(self._filename)

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
        if platform.system() in ['Linux','Darwin']:
            filename = '/etc/digits.cfg'
        else:
            filename = None
        super(SystemConfigFile, self).__init__(filename)

class UserConfigFile(ConfigFile):
    def __init__(self):
        if 'HOME' in os.environ:
            filename = os.path.join(os.environ['HOME'], '.digits.cfg')
        else:
            filename = None
        super(UserConfigFile, self).__init__(filename)

class InstanceConfigFile(ConfigFile):
    def __init__(self):
        filename = os.path.join(os.path.dirname(digits.__file__), 'digits.cfg')
        super(InstanceConfigFile, self).__init__(filename)

