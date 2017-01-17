# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from digits import utils
from digits.utils import subclass
from digits.utils.forms import validate_required_iff
from flask.ext.wtf import Form
import wtforms
from wtforms import validators


@subclass
class DatasetForm(Form):
    """
    A form used to create an image gradient dataset
    """

    train_image_count = utils.forms.IntegerField(
        'Train Image count',
        validators=[
            validators.DataRequired(),
            validators.NumberRange(min=1),
        ],
        default=1000,
        tooltip="Number of images to create in training set"
    )

    val_image_count = utils.forms.IntegerField(
        'Validation Image count',
        validators=[
            validators.Optional(),
            validators.NumberRange(min=0),
        ],
        default=250,
        tooltip="Number of images to create in validation set"
    )

    test_image_count = utils.forms.IntegerField(
        'Test Image count',
        validators=[
            validators.Optional(),
            validators.NumberRange(min=0),
        ],
        default=0,
        tooltip="Number of images to create in validation set"
    )

    image_width = wtforms.IntegerField(
        u'Image Width',
        default=32,
        validators=[validators.DataRequired()]
    )

    image_height = wtforms.IntegerField(
        u'Image Height',
        default=32,
        validators=[validators.DataRequired()]
    )


@subclass
class InferenceForm(Form):
    """
    A form used to perform inference on a gradient regression model
    """

    gradient_x = utils.forms.FloatField(
        'Gradient (x)',
        validators=[
            validate_required_iff(test_image_count=None),
            validators.NumberRange(min=-0.5, max=0.5),
        ],
        tooltip="Specify a number between -0.5 and 0.5"
    )

    gradient_y = utils.forms.FloatField(
        'Gradient (y)',
        validators=[
            validate_required_iff(test_image_count=None),
            validators.NumberRange(min=-0.5, max=0.5),
        ],
        tooltip="Specify a number between -0.5 and 0.5"
    )

    test_image_count = utils.forms.IntegerField(
        'Test Image count',
        validators=[
            validators.Optional(),
            validators.NumberRange(min=0),
        ],
        tooltip="Number of images to create in test set"
    )
