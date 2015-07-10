# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import wtforms
from wtforms import validators

from ..forms import DatasetForm
from job import ImageDatasetJob

class ImageDatasetForm(DatasetForm):
    """
    Defines the form used to create a new ImageDatasetJob
    (abstract class)
    """

    encoding = wtforms.SelectField('Image Encoding',
            default = 'png',
            choices = [
                ('none', 'None'),
                ('png', 'PNG (lossless)'),
                ('jpg', 'JPEG (lossy, 90% quality)'),
                ],
            )

    ### Image resize

    resize_channels = wtforms.SelectField(u'Image Type',
            default='3',
            choices=[('1', 'Grayscale'), ('3', 'Color')]
            )
    resize_width = wtforms.IntegerField(u'Resize Width',
            default=256,
            validators=[validators.DataRequired()]
            )
    resize_height = wtforms.IntegerField(u'Resize Height',
            default=256,
            validators=[validators.DataRequired()]
            )
    resize_mode = wtforms.SelectField(u'Resize Transformation',
            default='squash',
            choices=ImageDatasetJob.choices()
            )
