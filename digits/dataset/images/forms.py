# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import wtforms
from wtforms import validators

from ..forms import DatasetForm

class ImageDatasetForm(DatasetForm):
    """
    Defines the form used to create a new ImageDatasetJob
    (abstract class)
    """

    encode_images = wtforms.BooleanField('Save encoded JPEGs',
            default=True
            )

    ### Image resize

    resize_channels = wtforms.SelectField(u'Image type',
            default='3',
            choices=[('1', 'Grayscale'), ('3', 'Color')]
            )
    resize_width = wtforms.IntegerField(u'Resize width',
            default=256,
            validators=[validators.DataRequired()]
            )
    resize_height = wtforms.IntegerField(u'Resize height',
            default=256,
            validators=[validators.DataRequired()]
            )
    resize_mode = wtforms.SelectField(u'Resize transformation',
            default='half_crop',
            choices=[
                ('crop', 'Crop'),
                ('squash', 'Squash'),
                ('fill', 'Fill'),
                ('half_crop', 'Half crop, half fill'),
                ]
            )

