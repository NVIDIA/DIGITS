# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import re
import time

from digits import test_utils
from digits import utils


test_utils.skipIfNotFramework('none')


class TestTimeFilters():

    def test_print_time(self):

        # Pass in a reference time to print_time, to avoid stepping into the
        # next year in the other tests close to December.  Also avoid the
        # leap year and daylight savings time.
        t = (2009, 5, 17, 16, 0, 0, 0, 0, 0)
        ref_time = time.mktime(t)
        day = 60 * 60 * 24

        # Year test (365 days)
        s = utils.time_filters.print_time(day * 365 + ref_time, ref_time)
        assert re.match('\w{3} \d{2} \d{4}, \d{2}:\d{2}:\d{2} [AP]M', s)

        # Month test (40 days)
        s = utils.time_filters.print_time(day * 40 + ref_time, ref_time)
        assert re.match('\w{3} \d{2}, \d{2}:\d{2}:\d{2} [AP]M', s)

        # Day test (4 days)
        s = utils.time_filters.print_time(day * 4 + ref_time, ref_time)
        assert re.match('\w{3} \w{3} \d{2}, \d{2}:\d{2}:\d{2} [AP]M', s)

        # default test (4 seconds)
        s = utils.time_filters.print_time(4 + ref_time, ref_time)
        assert re.match('\d{2}:\d{2}:\d{2} [AP]M', s)

    def test_print_time_diff(self):

        def time_string(days, hours, minutes, seconds):
            time = 86400 * days + 3600 * hours + 60 * minutes + seconds
            return utils.time_filters.print_time_diff(time)

        # Test days and hours
        assert time_string(1, 0, 0, 0) == '1 day'
        assert time_string(1, 1, 0, 0) == '1 day, 1 hour'
        assert time_string(2, 1, 0, 0) == '2 days, 1 hour'
        assert time_string(1, 2, 0, 0) == '1 day, 2 hours'
        assert time_string(2, 2, 0, 0) == '2 days, 2 hours'

        # Test hours and minutes
        assert time_string(0, 1, 0, 0) == '1 hour'
        assert time_string(0, 1, 1, 0) == '1 hour, 1 minute'
        assert time_string(0, 2, 1, 0) == '2 hours, 1 minute'
        assert time_string(0, 1, 2, 0) == '1 hour, 2 minutes'
        assert time_string(0, 2, 2, 0) == '2 hours, 2 minutes'

        # Test minutes and seconds
        assert time_string(0, 0, 1, 0) == '1 minute'
        assert time_string(0, 0, 1, 1) == '1 minute, 1 second'
        assert time_string(0, 0, 2, 1) == '2 minutes, 1 second'
        assert time_string(0, 0, 1, 2) == '1 minute, 2 seconds'
        assert time_string(0, 0, 2, 2) == '2 minutes, 2 seconds'

        # Test seconds
        assert time_string(0, 0, 0, 1) == '1 second'
        assert time_string(0, 0, 0, 2) == '2 seconds'

        # Test no time
        assert time_string(0, 0, 0, 0) == '0 seconds'

        # Test negative time
        assert time_string(0, 0, 0, -2) == 'Negative Time'

        # Test None
        assert utils.time_filters.print_time_diff(None) == '?'

    def test_print_time_diff_nosuffixes(self):
        def time_string(hours, minutes, seconds):
            time = 3600 * hours + 60 * minutes + seconds
            return utils.time_filters.print_time_diff_nosuffixes(time)

        assert time_string(0, 0, 0) == '00:00:00'
        assert time_string(0, 0, 1) == '00:00:01'
        assert time_string(0, 1, 0) == '00:01:00'
        assert time_string(0, 1, 1) == '00:01:01'
        assert time_string(1, 0, 0) == '01:00:00'
        assert time_string(1, 0, 1) == '01:00:01'
        assert time_string(1, 1, 0) == '01:01:00'
        assert time_string(1, 1, 1) == '01:01:01'

        # Test None
        assert utils.time_filters.print_time_diff_nosuffixes(None) == '?'

    def test_print_time_since(self):
        assert utils.time_filters.print_time_since(time.time()) == '0 seconds'
