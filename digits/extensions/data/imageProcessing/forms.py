# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

from flask.ext.wtf import Form
from wtforms import validators

from digits import utils
from digits.utils import subclass
from digits.utils.forms import validate_required_iff


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
                raise validators.ValidationError(
                    'Folder does not exist or is not reachable')
            else:
                return True

    feature_folder = utils.forms.StringField(
        u'Feature image folder',
        validators=[
            validators.DataRequired(),
            validate_folder_path,
        ],
        tooltip="Indicate a folder full of images."
    )

    label_folder = utils.forms.StringField(
        u'Label image folder',
        validators=[
            validators.DataRequired(),
            validate_folder_path,
        ],
        tooltip="Indicate a folder full of images. For each image in the feature"
                " image folder there must be one corresponding image in the label"
                " image folder. The label image must have the same filename except"
                " for the extension, which may differ."
    )

    folder_pct_val = utils.forms.IntegerField(
        u'% for validation',
        default=10,
        validators=[
            validators.NumberRange(min=0, max=100)
        ],
        tooltip="You can choose to set apart a certain percentage of images "
                "from the training images for the validation set."
    )

    has_val_folder = utils.forms.BooleanField('Separate validation images',
                                              default=False,
                                              )

    validation_feature_folder = utils.forms.StringField(
        u'Validation feature image folder',
        validators=[
            validate_required_iff(has_val_folder=True),
            validate_folder_path,
        ],
        tooltip="Indicate a folder full of images."
    )

    validation_label_folder = utils.forms.StringField(
        u'Validation label image folder',
        validators=[
            validate_required_iff(has_val_folder=True),
            validate_folder_path,
        ],
        tooltip="Indicate a folder full of images. For each image in the feature"
                " image folder there must be one corresponding image in the label"
                " image folder. The label image must have the same filename except"
                " for the extension, which may differ."
    )

    channel_conversion = utils.forms.SelectField(
        'Channel conversion',
        choices=[
            ('RGB', 'RGB'),
            ('L', 'Grayscale'),
            ('none', 'None'),
        ],
        default='none',
        tooltip="Perform selected channel conversion."
    )
