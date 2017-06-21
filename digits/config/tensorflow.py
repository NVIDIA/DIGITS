# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import platform
from subprocess import Popen, PIPE

from . import option_list

VARNAME_ENV_TFPY = 'TENSORFLOW_PYTHON'
DEFAULT_PYTHON_EXE = 'python2'  # @TODO(tzaman) - use the python executable that was used to launch digits?

if platform.system() == 'Darwin':
    # DYLD_LIBRARY_PATH and LD_LIBRARY_PATH is sometimes stripped, and the cuda libraries might need it
    if "DYLD_LIBRARY_PATH" not in os.environ:
        if "CUDA_HOME" in os.environ:
            os.environ["DYLD_LIBRARY_PATH"] = str(os.environ["CUDA_HOME"] + '/lib')


def test_tf_import(python_exe):
    """
    Tests if tensorflow can be imported, returns if it went okay and optional error.
    """
    p = Popen([python_exe, "-c", "import tensorflow"], stdout=PIPE, stderr=PIPE)
    (out, err) = p.communicate()
    return p.returncode == 0, str(err)

if VARNAME_ENV_TFPY in os.environ:
    tf_python_exe = os.environ[VARNAME_ENV_TFPY]
else:
    tf_python_exe = DEFAULT_PYTHON_EXE

tf_enabled, err = test_tf_import(tf_python_exe)

if not tf_enabled:
    print('Tensorflow support disabled.')
#   print('Failed importing Tensorflow with python executable "%s"\n%s' % (tf_python_exe, err))

if tf_enabled:
    option_list['tensorflow'] = {
        'enabled': True,
        'executable': tf_python_exe,
    }
else:
    option_list['tensorflow'] = {
        'enabled': False,
    }
