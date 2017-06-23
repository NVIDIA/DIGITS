# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import unittest

from . import parse_version
from digits import test_utils


test_utils.skipIfNotFramework('none')


class TestParseVersion():

    def test_equality(self):
        for v1, v2 in [
                ('11', '11'),
                ('11', 11),
                ('11', ('11',)),
                ('11', (11,)),
                ('11', '11.0'),
                ('11', '11.0.0'),
        ]:
            yield self.check_equal, v1, v2

    def check_equal(self, v1, v2):
        assert parse_version(v1) == parse_version(v2)

    def test_lt(self):
        # Each list should be in strictly increasing order
        example_lists = []

        # Some DIGITS versions
        example_lists.append(
            'v1.0 v1.0.1 v1.0.2 v1.0.3 v1.1.0-rc1 v1.1.0-rc2 v1.1.0 v1.1.1 '
            'v1.1.2 v1.1.3 v2.0.0-rc v2.0.0-rc2 v2.0.0-rc3 v2.0.0-preview '
            'v2.0.0 v2.1.0 v2.2.0 v2.2.1'.split()
        )
        # Some NVcaffe versions
        example_lists.append('v0.13.0 v0.13.1 v0.13.2 v0.14.0-alpha '
                             'v0.14.0-beta v0.14.0-rc.1 v0.14.0-rc.2'.split())
        # Semver.org examples
        example_lists.append(
            '1.0.0-alpha 1.0.0-alpha.1 1.0.0-alpha.beta 1.0.0-beta '
            '1.0.0-beta.2 1.0.0-beta.11 1.0.0-rc.1 1.0.0'.split()
        )
        # PEP 440 examples
        example_lists.append('0.1 0.2 0.3 1.0 1.1'.split())
        example_lists.append('1.1.0 1.1.1 1.1.2 1.2.0'.split())
        example_lists.append('0.9 1.0a1 1.0a2 1.0b1 1.0rc1 1.0 1.1a1'.split())
        example_lists.append('0.9 1.0.dev1 1.0.dev2 1.0.dev3 1.0.dev4 1.0c1 1.0c2 1.0 1.0.post1 1.1.dev1'.split())
        example_lists.append('2012.1 2012.2 2012.3 2012.15 2013.1 2013.2'.split())

        for l in example_lists:
            for v1, v2 in zip(l[:-1], l[1:]):
                yield self.check_lt, v1, v2

    def check_lt(self, v1, v2):
        bad = (
            # pkg_resources handles this one differently
            '1.0.0-alpha.beta',
            # poor decision
            'v2.0.0-preview')
        if v1 in bad or v2 in bad:
            raise unittest.case.SkipTest
        assert parse_version(v1) < parse_version(v2)
