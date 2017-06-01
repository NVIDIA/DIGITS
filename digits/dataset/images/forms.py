# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import wtforms
from wtforms import validators

from ..forms import DatasetForm
from .job import ImageDatasetJob
from digits import utils


class ImageDatasetForm(DatasetForm):
    """
    Defines the form used to create a new ImageDatasetJob
    (abstract class)
    """

    encoding = utils.forms.SelectField(
        'Image Encoding',
        default='png',
        choices=[
            ('none', 'None'),
            ('png', 'PNG (lossless)'),
            ('jpg', 'JPEG (lossy, 90% quality)'),
        ],
        tooltip=('Using either of these compression formats can save disk space, '
                 'but can also require marginally more time for training.'),
    )

    # Image resize

    resize_channels = utils.forms.SelectField(
        u'Image Type',
        default='3',
        choices=[('1', 'Grayscale'), ('3', 'Color')],
        tooltip="Color is 3-channel RGB. Grayscale is single channel monochrome."
    )
    resize_width = wtforms.IntegerField(
        u'Resize Width',
        default=256,
        validators=[validators.DataRequired()]
    )
    resize_height = wtforms.IntegerField(
        u'Resize Height',
        default=256,
        validators=[validators.DataRequired()]
    )
    resize_mode = utils.forms.SelectField(
        u'Resize Transformation',
        default='squash',
        choices=ImageDatasetJob.resize_mode_choices(),
        tooltip="Options for dealing with aspect ratio changes during resize. See examples below."
    )
