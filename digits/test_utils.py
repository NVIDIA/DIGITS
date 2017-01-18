# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
"""
Utility functions used in other test files
"""
from __future__ import absolute_import

import os
import unittest

from digits.config import config_value


def skipIfNotFramework(framework):
    """
    Raises SkipTest if DIGITS_TEST_FRAMEWORK is set
    to something other than framework
    """
    key = 'DIGITS_TEST_FRAMEWORK'
    if (key in os.environ and os.environ[key] != framework):
        raise unittest.SkipTest(
            'Skipping because %s is "%s" and not "%s"'
            % (key, os.environ[key], framework))


class DatasetMixin(object):
    """
    Mixin for dataset tests - skip if framework is not "none"
    """
    @classmethod
    def setUpClass(cls):
        skipIfNotFramework('none')

        # Call super.setUpClass() unless we're the last in the class hierarchy
        supercls = super(DatasetMixin, cls)
        if hasattr(supercls, 'setUpClass'):
            supercls.setUpClass()


class CaffeMixin(object):
    """
    Mixin for caffe tests
    """
    FRAMEWORK = 'caffe'

    @classmethod
    def setUpClass(cls):
        skipIfNotFramework('caffe')

        # Call super.setUpClass() unless we're the last in the class hierarchy
        supercls = super(CaffeMixin, cls)
        if hasattr(supercls, 'setUpClass'):
            supercls.setUpClass()


class TorchMixin(object):
    """
    Mixin for torch tests
    """
    FRAMEWORK = 'torch'

    @classmethod
    def setUpClass(cls):
        skipIfNotFramework('torch')
        if cls.FRAMEWORK == 'torch' and not config_value('torch')['enabled']:
            raise unittest.SkipTest('Torch not found')

        # Call super.setUpClass() unless we're the last in the class hierarchy
        supercls = super(TorchMixin, cls)
        if hasattr(supercls, 'setUpClass'):
            supercls.setUpClass()
