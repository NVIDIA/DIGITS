# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from . import imageGradients
from . import imageProcessing
from . import imageSegmentation
from . import objectDetection

data_extensions = [
    # Set show=True if extension should be shown by default
    # on DIGITS home page. These defaults can be changed by
    # editing DIGITS config option 'data_extension_list'
    {'class': imageGradients.DataIngestion, 'show': False},
    {'class': imageProcessing.DataIngestion, 'show': True},
    {'class': imageSegmentation.DataIngestion, 'show': True},
    {'class': objectDetection.DataIngestion, 'show': True},
]


def get_extensions(show_all=False):
    """
    return set of data data extensions
    """
    return [extension['class']
            for extension in data_extensions
            if show_all or extension['show']]


def get_extension(extension_id):
    """
    return extension associated with specified extension_id
    """
    for extension in data_extensions:
        extension_class = extension['class']
        if extension_class.get_id() == extension_id:
            return extension_class
    return None
