#!/usr/bin/env python
# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import os
import argparse

import config_option
import config_file
import prompt
import current_config

def print_config(verbose=False):
    """
    Prints out a matrix of config option values for each level
    """
    min_visibility = config_option.Visibility.DEFAULT
    if verbose:
        min_visibility = config_option.Visibility.HIDDEN

    levels = [
            ('INSTANCE', config_file.InstanceConfigFile()),
            ('USER', config_file.UserConfigFile()),
            ('SYSTEM', config_file.SystemConfigFile())]
    # filter out the files which don't exist
    levels = [l for l in levels if l[1].can_read()]

    if len(levels) == 0:
        # nothing to display
        return None

    # create a row for each option
    row_headers = []
    row_data = []
    for option in [o for o in current_config.option_list
            if o.visibility() >= min_visibility]:
        row_headers.append(option.prompt_title())
        row = []
        for title, config in levels:
            value = config.get(option.config_file_key())
            row.append(prompt.value_to_str(value))
        row_data.append(row)

    prompt.print_section_header('Current Config')

    # calculate the width of each column for pretty printing
    row_header_width = max([len(h) for h in row_headers])

    row_data_widths = []
    for i, level in enumerate(levels):
        title, config = level
        w = len(title)
        for row in row_data:
            if len(row[i]) > w:
                w = len(row[i])
        row_data_widths.append(w)

    # build the format string for printing
    row_format = '%%%ds' % row_header_width
    for width in row_data_widths:
        row_format += ' | %%-%ds' % width

    # print header row
    print row_format % (('',) + tuple([level[0] for level in levels]))

    # print option rows
    for i, row in enumerate(row_data):
        print row_format % ((row_headers[i],) + tuple(row))
    print


def edit_config_file(verbose=False):
    """
    Prompt the user for which file to edit,
    then allow them to set options in that file
    """
    suggestions = []
    instanceConfig = config_file.InstanceConfigFile()
    if instanceConfig.can_write():
        suggestions.append(prompt.Suggestion(
            instanceConfig.filename(), 'I',
            desc = 'Instance', default=True))
    userConfig = config_file.UserConfigFile()
    if userConfig.can_write():
        suggestions.append(prompt.Suggestion(
            userConfig.filename(), 'U',
            desc = 'User', default=True))
    systemConfig = config_file.SystemConfigFile()
    if systemConfig.can_write():
        suggestions.append(prompt.Suggestion(
            systemConfig.filename(), 'S',
            desc = 'System', default=True))

    def filenameValidator(filename):
        """
        Returns True if this is a valid file to edit
        """
        if os.path.isfile(filename):
            if not os.access(filename, os.W_OK):
                raise config_option.BadValue('You do not have write permission')
            else:
                return filename

        if os.path.isdir(filename):
            raise config_option.BadValue('This is a directory')
        dirname = os.path.dirname(os.path.realpath(filename))
        if not os.path.isdir(dirname):
            raise config_option.BadValue('Path not found: %s' % dirname)
        elif not os.access(dirname, os.W_OK):
            raise config_option.BadValue('You do not have write permission')
        return filename

    filename = prompt.get_input(
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

    configFile = config_file.ConfigFile(filename)

    min_visibility = config_option.Visibility.DEFAULT
    if verbose:
        min_visibility = config_option.Visibility.HIDDEN

    # Loop through the visible options
    for option in [o for o in current_config.option_list
            if o.visibility() >= min_visibility]:
        previous_value = configFile.get(option.config_file_key())
        suggestions = [prompt.Suggestion(None, 'U',
            desc='unset', default=(previous_value is None))]
        if previous_value is not None:
            suggestions.append(prompt.Suggestion(previous_value, '',
                desc = 'Previous', default = True))
        if instanceConfig is not None:
            instance_value = instanceConfig.get(option.config_file_key())
            if instance_value is not None:
                suggestions.append(prompt.Suggestion(instance_value, 'I',
                    desc = 'Instance', default = is_standard_location))
        if userConfig is not None:
            user_value = userConfig.get(option.config_file_key())
            if user_value is not None:
                suggestions.append(prompt.Suggestion(user_value, 'U',
                    desc = 'User', default = is_standard_location))
        if systemConfig is not None:
            system_value = systemConfig.get(option.config_file_key())
            if system_value is not None:
                suggestions.append(prompt.Suggestion(system_value, 'S',
                    desc = 'System', default = is_standard_location))
        suggestions += option.suggestions()
        if option.optional():
            suggestions.append(prompt.Suggestion('', 'N',
                desc = 'none', default = True))

        prompt.print_section_header(option.prompt_title())
        value = prompt.get_input(
                message     = option.prompt_message(),
                validator   = option.validate,
                suggestions = suggestions,
                is_path     = option.is_path(),
                )
        print
        configFile.set(option.config_file_key(), value)

    configFile.save()
    print 'New config saved at %s' % configFile.filename()
    print
    print configFile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Config - DIGITS')
    parser.add_argument('-v', '--verbose',
            action="store_true",
            help='view more options')

    args = vars(parser.parse_args())

    print_config(args['verbose'])
    edit_config_file(args['verbose'])

