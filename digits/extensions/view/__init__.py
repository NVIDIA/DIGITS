# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from . import boundingBox
from . import imageOutput
from . import imageGradients
from . import imageSegmentation
from . import rawData


view_extensions = [
    # Set show=True if extension should be shown by default
    # in the 'Select Visualization Method' dialog. These defaults
    # can be changed by editing DIGITS config option
    # 'view_extension_list'
    {'class': boundingBox.Visualization, 'show': True},
    {'class': imageGradients.Visualization, 'show': False},
    {'class': imageOutput.Visualization, 'show': True},
    {'class': imageSegmentation.Visualization, 'show': True},
    {'class': rawData.Visualization, 'show': True},
]


def get_default_extension():
    """
    return the default view extension
    """
    return rawData.Visualization


def get_extensions(show_all=False):
    """
    return set of data data extensions
    """
    return [extension['class']
            for extension in view_extensions
            if show_all or extension['show']]


def get_extension(extension_id):
    """
    return extension associated with specified extension_id
    """
    for extension in view_extensions:
        extension_class = extension['class']
        if extension_class.get_id() == extension_id:
            return extension_class
    return None
