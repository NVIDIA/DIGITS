from __future__ import absolute_import

from . import option_list
import subprocess
import os
from digits.extensions.cluster_management.slurm import test_if_slurm_system

if test_if_slurm_system():
    system_type = 'int'
else:
    system_type = 'interactive'
option_list['system_type'] = system_type
