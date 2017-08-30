# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from flask.ext.wtf import Form

from digits import utils
from digits.utils import subclass


@subclass
class ConfigForm(Form):
    """
    A form used to display the network output as an image
    """
    colormap = utils.forms.SelectField(
        'Colormap',
        choices=[
            ('dataset', 'From dataset'),
            ('paired', 'Paired (matplotlib)'),
            ('none', 'None (grayscale)'),
        ],
        default='dataset',
        tooltip='Set color map to use when displaying segmented image'
    )
