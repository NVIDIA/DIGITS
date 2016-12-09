from __future__ import absolute_import

from . import option_list
import subprocess
import os

try:
    if subprocess.call('slurm',  stdout=subprocess.PIPE) == 0:
        system_type = "slurm"
        os.environ['TMPDIR'] = str(os.path.abspath('./tmp'))
    else:
        system_type = "int"
except OSError:
    system_type = "int"
option_list['system_type'] = system_type
