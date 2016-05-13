# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from digits import utils
from digits.utils import subclass
from flask.ext.wtf import Form
import wtforms
from wtforms import validators


@subclass
class ConfigForm(Form):
    """
    A form used to configure the drawing of bounding boxes
    """

    box_color = wtforms.SelectField(
        'Box color',
        choices=[
            ('red', 'Red'),
            ('green', 'Green'),
            ('blue', 'Blue'),
            ],
        default='red',
        )

    line_width = utils.forms.IntegerField(
        'Line width',
        validators=[
            validators.DataRequired(),
            validators.NumberRange(min=1),
            ],
        default=2,
        )
