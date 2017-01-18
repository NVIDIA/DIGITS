# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import tempfile

from . import option_list
import digits


if 'DIGITS_MODE_TEST' in os.environ:
    value = tempfile.mkdtemp()
elif 'DIGITS_JOBS_DIR' in os.environ:
    value = os.environ['DIGITS_JOBS_DIR']
else:
    value = os.path.join(os.path.dirname(digits.__file__), 'jobs')


try:
    value = os.path.abspath(value)
    if os.path.exists(value):
        if not os.path.isdir(value):
            raise IOError('No such directory: "%s"' % value)
        if not os.access(value, os.W_OK):
            raise IOError('Permission denied: "%s"' % value)
    if not os.path.exists(value):
        os.makedirs(value)
except:
    print '"%s" is not a valid value for jobs_dir.' % value
    print 'Set the envvar DIGITS_JOBS_DIR to fix your configuration.'
    raise


option_list['jobs_dir'] = value
