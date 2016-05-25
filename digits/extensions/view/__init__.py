# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from . import boundingBox
from . import rawData

view_extensions = [
    # set show=True if extension should be listed in known extensions
    {'class': boundingBox.Visualization, 'show': True},
    {'class': rawData.Visualization, 'show': True},
]


def get_default_extension():
    """
    return the default view extension
    """
    return rawData.Visualization


def get_extensions():
    """
    return set of data data extensions
    """
    return [extension['class'] for extension
            in view_extensions if extension['show']]


def get_extension(extension_id):
    """
    return extension associated with specified extension_id
    """
    for extension in view_extensions:
        extension_class = extension['class']
        if extension_class.get_id() == extension_id:
            return extension_class
    return None
