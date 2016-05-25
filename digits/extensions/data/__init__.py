# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from . import imageGradients
from . import objectDetection

data_extensions = [
    # set show=True if extension should be listed in known extensions
    {'class': imageGradients.DataIngestion, 'show': False},
    {'class': objectDetection.DataIngestion, 'show': True},
]


def get_extensions():
    """
    return set of data data extensions
    """
    return [extension['class'] for extension in data_extensions if extension['show']]


def get_extension(extension_id):
    """
    return extension associated with specified extension_id
    """
    for extension in data_extensions:
        extension_class = extension['class']
        if extension_class.get_id() == extension_id:
            return extension_class
    return None
