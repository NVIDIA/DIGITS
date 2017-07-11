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

    data_order = utils.forms.SelectField(
        'Data order',
        choices=[
            ('chw', 'CHW'),
            ('hwc', 'HWC'),
            ],
        default='chw',
        tooltip="Set the order of the data. For Caffe and Torch models this "
                "is often NCHW, for Tensorflow it's NHWC."
                "N=Batch Size, W=Width, H=Height, C=Channels"
        )

    pixel_conversion = utils.forms.SelectField(
        'Pixel conversion',
        choices=[
            ('normalize', 'Normalize'),
            ('clip', 'Clip'),
        ],
        default='normalize',
        tooltip='Select method to convert pixel values to the target bit '
                'range'
    )

    show_input = utils.forms.SelectField(
        'Show input as image',
        choices=[
            ('yes', 'Yes'),
            ('no', 'No'),
        ],
        default='no',
        tooltip='Show input as image'
    )
