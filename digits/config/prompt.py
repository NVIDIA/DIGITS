# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

"""
Classes and functions relating to prompting a user for configuration options
"""

import sys
import os.path
import readline

import config_option

def print_section_header(title):
    """
    Utility for printing a section header
    """
    print '{s:{c}^{n}}'.format(
            s = ' %s ' % title,
            # Extend to 80 characters
            n = 80, c = '=')

def value_to_str(value):
    if value is None:
        return ''
    elif type(value) is not str:
        return str(value)
    elif not value.strip():
        return '<NONE>'
    else:
        return value

class Suggestion(object):
    """
    A simple class for Option suggested values (used in get_input())
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
                        except config_option.BadValue as e:
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
                            except config_option.BadValue as e:
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
                except config_option.BadValue as e:
                    print 'ERROR:', e

        if not valid:
            print 'Invalid input'

    if is_path:
        # back to normal
        readline.set_completer_delims(delims)
        readline.parse_and_bind('TAB: ')

    return value
