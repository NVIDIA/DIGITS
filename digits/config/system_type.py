from __future__ import absolute_import

from . import option_list
import subprocess

try:
    if subprocess.call('slurm',  stdout=subprocess.PIPE) == 0:
        system_type = "slurm"
    else:
        system_type = "int"
except OSError:
    system_type = "int"
option_list['system_type'] = system_type
