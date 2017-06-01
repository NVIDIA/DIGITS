# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.

import mock
import tempfile

from . import resize_image
from digits import test_utils


test_utils.skipIfNotFramework('none')


class TestOutputValidation():

    def test_no_filename(self):
        assert resize_image.validate_output_file(None), 'All new files should be valid'

    @mock.patch('os.access')
    def test_not_writable(self, mock_access):
        mock_access.return_value = False
        with tempfile.NamedTemporaryFile('r') as f:
            assert not resize_image.validate_output_file(f.name), 'validation should not pass on unwritable file'

    def test_normal(self):
        with tempfile.NamedTemporaryFile('r') as f:
            assert resize_image.validate_output_file(f.name), 'validation should pass on temporary file'


class TestInputValidation():

    def test_does_not_exist(self):
        assert not resize_image.validate_input_file(''), 'validation should not pass on missing file'

    @mock.patch('os.access')
    def test_unreadable_file(self, mock_access):
        mock_access.return_value = False
        with tempfile.NamedTemporaryFile('r') as f:
            assert not resize_image.validate_input_file(f.name), 'validation should not pass on unreadable file'


class TestRangeValidation():

    def test_number_none_and_not_allowed(self):
        assert not resize_image.validate_range(
            None, allow_none=False), 'number=None should not be allowed with allow_none=False'

    def test_number_not_float_compatible(self):
        value = 'a'
        assert not resize_image.validate_range(value), 'number=%s should not be accepted' % value

    def test_number_below_min(self):
        assert not resize_image.validate_range(0, min_value=1), 'validation should not pass with number < min_value'

    def test_number_above_max(self):
        assert not resize_image.validate_range(2, max_value=1), 'validation should not pass with number > max_value'

    def test_range(self):
        assert resize_image.validate_range(
            5, min_value=0, max_value=255), 'validation should pass with 5 in range (0, 255)'
