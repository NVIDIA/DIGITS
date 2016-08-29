# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from flask.ext.wtf import Form
from wtforms import validators

from digits import utils
from digits.utils import subclass


@subclass
class ConfigForm(Form):
    """
    A form used to display the network output as an image
    """
    channel_order = utils.forms.SelectField(
        'Channel order',
        choices=[
            ('rgb', 'RGB'),
            ('bgr', 'BGR'),
            ],
        default='rgb',
        tooltip='Set channel order to BGR for Caffe networks (this field '
                'is ignored in the case of a grayscale image)'
        )

    pixel_conversion = utils.forms.SelectField(
        'Pixel conversion',
        choices=[
            ('normalize', 'Normalize'),
            ('clip', 'Clip'),
            ('threshold', 'Binary threshold')
            ],
        default='normalize',
        tooltip='Select method to convert pixel values to the target bit '
                'range'
        )

    threshold_val = utils.forms.IntegerField(
        u'Threshold',
        default=128,
        validators=[
            validators.NumberRange(min=0, max=255)
            ],
        tooltip="Threshold value"
        )
