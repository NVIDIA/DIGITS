from __future__ import absolute_import
from . import option_list
from digits.extensions.cluster_management.cluster_factory import cluster_factory
if cluster_factory.use_cluster:
    system_type = cluster_factory.selected_system

else:
    system_type = 'interactive'

option_list['system_type'] = system_type
