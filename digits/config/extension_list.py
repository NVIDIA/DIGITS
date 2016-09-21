# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

from . import option_list
from digits import extensions


if 'DIGITS_ALL_EXTENSIONS' in os.environ:
    option_list['data_extension_list'] = extensions.data.get_extensions(show_all=True)
    option_list['view_extension_list'] = extensions.view.get_extensions(show_all=True)
else:
    option_list['data_extension_list'] = extensions.data.get_extensions(show_all=False)
    option_list['view_extension_list'] = extensions.view.get_extensions(show_all=False)
