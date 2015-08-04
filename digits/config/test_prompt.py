# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import sys
from contextlib import contextmanager
from StringIO import StringIO

import mock
from nose.tools import raises

from . import prompt as _

class TestValueToStr():
    def test_none(self):
        # pass none to value_to_str
        assert _.value_to_str(None) == '', 'passing None should return an empty string'

    def test_nonstring(self):
        # pass a non-string value to value_to_str
        assert _.value_to_str(1) == '1', 'passing 1 should return the string "1"'

class TestSuggestion():
    @raises(ValueError)
    def test_new_bad_char_type(self):
        # pass a non-string type as char to suggestion
        _.Suggestion(None, 1)

    @raises(ValueError)
    def test_new_bad_multichar(self):
        # pass multiple chars where one is expected
        _.Suggestion(None, 'badvalue')

    def test_str_method(self):
        # test __str__ method of Suggestion
        suggestion = _.Suggestion('alpha', 'a', 'test', True)
        strval = str(suggestion)
        expect = '<Suggestion char="a" desc="test" value="alpha" default>'

        assert strval == expect, 'Suggestion is not producing the correct string value %s' % expect

@contextmanager
def mockInput(fn):
    original = __builtins__['raw_input']
    __builtins__['raw_input'] = fn
    yield
    __builtins__['raw_input'] = original

class TestGetInput():
    def setUp(self):
        self.suggestions = [_.Suggestion('alpha', 'a', 'test', False)]

    @raises(SystemExit)
    def test_get_input_sys_exit(self):
        # bad input from user
        def temp(_):
            raise KeyboardInterrupt

        with mockInput(temp):
            _.get_input('Test', lambda _: True, self.suggestions)

    def test_get_input_empty_then_full(self):
        # test both major paths of get_input
        # Python 2 does not have the 'nonlocal' keyword, so we fudge the closure with an object.
        class Temp:
            def __init__(self):
                self.flag = False
            def __call__(self, _):
                if not self.flag:
                    self.flag = True
                    return ''
                else:
                    return 'a'

        with mockInput(Temp()):
            assert _.get_input('Test', lambda x: x, self.suggestions) == 'alpha', 'get_input should return "alpha" for input "a"'

    def test_get_input_empty_default(self):
        # empty input should choose the default
        self.suggestions[0].default = True

        with mockInput(lambda _: ''):
            assert _.get_input('Test', lambda x: x+'_validated', self.suggestions) == 'alpha_validated', 'get_input should return the default value "alpha"'

    def test_get_input_empty_default_no_validator(self):
        # empty input should choose the default and not validate
        self.suggestions[0].default = True

        with mockInput(lambda _: ''):
            assert _.get_input('Test', suggestions=self.suggestions) == 'alpha', 'get_input should return the default value "alpha"'

    @mock.patch('os.path.expanduser')
    def test_get_input_path(self, mock_expanduser):
        # should correctly validate path
        mock_expanduser.side_effect = lambda x: '/path'+x

        with mockInput(lambda _: '/test'):
            assert _.get_input(validator=lambda x: x, is_path=True) == '/path/test', 'get_input should return the default value "alpha"'

