# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import copy
from pkg_resources import iter_entry_points

from . import imageProcessing
from . import imageSegmentation
from . import objectDetection

# Entry point group (this is the key we use to register and
# find installed plug-ins)
GROUP = "digits.plugins.data"

# built-in extensions
builtin_data_extensions = [
    imageProcessing.DataIngestion,
    imageSegmentation.DataIngestion,
    objectDetection.DataIngestion,
]


def get_extensions():
    """
    return set of data data extensions
    """
    extensions = copy.copy(builtin_data_extensions)
    # find installed extension plug-ins
    for entry_point in iter_entry_points(group=GROUP, name=None):
        extensions.append(entry_point.load())

    return extensions


def get_extension(extension_id):
    """
    return extension associated with specified extension_id
    """
    for extension in get_extensions():
        if extension.get_id() == extension_id:
            return extension
    return None
