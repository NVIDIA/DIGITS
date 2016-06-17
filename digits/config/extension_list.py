# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from digits import extensions
from . import config_option
from . import prompt


class ExtensionListOption(config_option.Option):
    """
    Common interface for extension list options
    """
    @classmethod
    def prompt_message(cls):
        s = 'Available extensions:\n'
        for extension in cls.get_extensions(True):
            s += '\tID=\'%s\' Title=\'%s\'\n' % (extension.get_id(), extension.get_title())
        s += '\nInput the IDs of the extensions you would like to use, separated by commas.'
        return s

    def optional(self):
        return True

    def suggestions(self):
        if len(self.get_extensions(False)) > 0:
            return [
                prompt.Suggestion(
                    ','.join([ext.get_id() for ext in self.get_extensions(False)]),
                    'D', desc='default', default=True),
                prompt.Suggestion(
                    ','.join([ext.get_id() for ext in self.get_extensions(True)]),
                    'A', desc='all', default=False),
                ]
        else:
            return []

    @classmethod
    def visibility(cls):
        if len(cls.get_extensions(True)) == 0:
            # Nothing to see here
            return config_option.Visibility.NEVER
        else:
            return config_option.Visibility.DEFAULT

    @classmethod
    def validate(cls, value):
        if value == '':
            return value

        choices = []
        extensions = cls.get_extensions(True)

        if not len(extensions):
            return ''
        if len(extensions) and not value.strip():
            raise config_option.BadValue('Empty list')
        for word in value.split(','):
            if not word:
                continue
            if not cls.get_extension(word):
                raise config_option.BadValue('There is no extension with ID=`%s`' % word)
            if word in choices:
                raise config_option.BadValue('You cannot select an extension twice')
            choices.append(word)

        if len(choices) > 0:
            return ','.join(choices)
        else:
            raise config_option.BadValue('Empty list')

    def _set_config_dict_value(self, value):
        """
        Set _config_dict_value according to a validated value
        """
        extensions = []
        for word in value.split(','):
            extension = self.get_extension(word)
            if extension is not None:
                extensions.append(extension)
        self._config_dict_value = extensions


class DataExtensionListOption(ExtensionListOption):
    """
    Extension list sub-class for data extensions
    """
    @staticmethod
    def config_file_key():
        return 'data_extension_list'

    @classmethod
    def prompt_title(cls):
        return 'Data extensions'

    @classmethod
    def get_extension(cls, extension_id):
        return extensions.data.get_extension(extension_id)

    @classmethod
    def get_extensions(cls, show_all):
        return extensions.data.get_extensions(show_all=show_all)


class ViewExtensionListOption(ExtensionListOption):
    """
    Extension list sub-class for data extensions
    """
    @staticmethod
    def config_file_key():
        return 'view_extension_list'

    @classmethod
    def prompt_title(cls):
        return 'View extensions'

    @classmethod
    def get_extension(cls, extension_id):
        return extensions.view.get_extension(extension_id)

    @classmethod
    def get_extensions(cls, show_all):
        return extensions.view.get_extensions(show_all=show_all)
