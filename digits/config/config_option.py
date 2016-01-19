# Copyright (c) 2015-2016, NVIDIA CORPORATION.  All rights reserved.

class BadValue(Exception):
    # thrown when a bad value is passed to option.set()
    pass

class Visibility(object):
    NEVER, HIDDEN, DEFAULT = range(3)

class Option(object):
    """
    Base class for configuration options
    """
    def __init__(self):
        self._valid = False
        self._config_file_value = None
        self._config_dict_value = None

    @staticmethod
    def config_file_key():
        """
        Key in the config file for this option
        """
        raise NotImplementedError

    @classmethod
    def prompt_title(cls):
        """
        Title to print for prompt
        """
        return cls.config_file_key()

    @classmethod
    def prompt_message(cls):
        """
        Message to print for prompt
        """
        return None

    @classmethod
    def visibility(cls):
        return Visibility.DEFAULT

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

    @staticmethod
    def is_path():
        """
        If True, tab autocompletion will be turned on during prompt
        """
        return False

    @staticmethod
    def has_test_value():
        """
        If true, use test_value during testing
        """
        return False

    @staticmethod
    def test_value():
        """
        Returns a special value to be used during testing
        Ignores the current configuration
        """
        raise NotImplementedError

    def valid(self):
        """
        Returns True if this option has been set with a valid value
        """
        return self._valid

    def has_value(self):
        """
        Returns False if value is either None or ''
        """
        return self.valid() and bool(self._config_file_value)

    @classmethod
    def validate(cls, value):
        """
        Returns a fixed-up valid version of value
        Raises BadValue if invalid
        """
        return value

    def set(self, value):
        """
        Set the value
        Raises BadValue
        """
        value = self.validate(value)
        self._config_file_value = value
        self._set_config_dict_value(value)
        self._valid = True

    def _set_config_dict_value(self, value):
        """
        Set _config_dict_value according to a validated value
        You may want to override this to store more detailed information
        """
        self._config_dict_value = value

    def config_dict_value(self):
        return self._config_dict_value

    def apply(self):
        """
        Apply this configuration
        (may involve altering the PATH)
        """
        pass

class FrameworkOption(Option):
    """
    Base class for DL framework backends
    """
    def optional(self):
        return True

