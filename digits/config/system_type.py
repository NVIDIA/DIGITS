from __future__ import absolute_import

from . import option_list
import subprocess


if subprocess.call('slurm', stdout=None) == 0:
    system_type = "slurm"
else:
    system_type = "int"

option_list['system_type'] = system_type