# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from flask.ext.wtf import Form
import os
from wtforms import validators

from digits import utils
from digits.utils import subclass


@subclass
class DatasetForm(Form):
    """
    A form used to create an image processing dataset
    """

    def validate_folder_path(form, field):
        if not field.data:
            pass
        else:
            # make sure the filesystem path exists
            if not os.path.exists(field.data) or not os.path.isdir(field.data):
                raise validators.ValidationError('Folder does not exist or is not reachable')
            else:
                return True

    train_image_folder = utils.forms.StringField(
        u'Training image folder',
        validators=[
            validators.DataRequired(),
            validate_folder_path,
            ],
        tooltip="Indicate a folder of images to use for training"
        )

    train_label_folder = utils.forms.StringField(
        u'Training label folder',
        validators=[
            validators.DataRequired(),
            validate_folder_path,
            ],
        tooltip="Indicate a folder of training labels"
        )

    val_image_folder = utils.forms.StringField(
        u'Validation image folder',
        validators=[
            validators.Optional(),
            validate_folder_path,
            ],
        tooltip="Indicate a folder of images to use for training"
        )

    val_label_folder = utils.forms.StringField(
        u'Validation label folder',
        validators=[
            validators.Optional(),
            validate_folder_path,
            ],
        tooltip="Indicate a folder of validation labels"
        )

    resize_image_width = utils.forms.IntegerField(
        u'Resize Image Width',
        validators=[
            validators.Optional(),
            validators.NumberRange(min=1),
            ],
        tooltip="If specified, images will be resized to that dimension after padding"
        )

    resize_image_height = utils.forms.IntegerField(
        u'Resize Image Height',
        validators=[
            validators.Optional(),
            validators.NumberRange(min=1),
            ],
        tooltip="If specified, images will be resized to that dimension after padding"
        )

    padding_image_width = utils.forms.IntegerField(
        u'Padding Image Width',
        default=1248,
        validators=[
            validators.DataRequired(),
            validators.NumberRange(min=1),
            ],
        tooltip="Images will be padded to that dimension"
        )

    padding_image_height = utils.forms.IntegerField(
        u'Padding Image Height',
        default=384,
        validators=[
            validators.DataRequired(),
            validators.NumberRange(min=1),
            ],
        tooltip="Images will be padded to that dimension"
        )

    channel_conversion = utils.forms.SelectField(
        'Channel conversion',
        choices=[
            ('RGB', 'RGB'),
            ('L', 'Grayscale'),
            ('none', 'None'),
            ],
        default='RGB',
        tooltip="Perform selected channel conversion."
        )
