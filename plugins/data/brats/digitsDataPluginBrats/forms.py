# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

from digits import utils
from digits.utils import subclass
from flask.ext.wtf import Form
from wtforms import validators


@subclass
class DatasetForm(Form):
    """
    A form used to create a Sunnybrook dataset
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

    dataset_folder = utils.forms.StringField(
        u'Dataset folder',
        validators=[
            validators.DataRequired(),
            validate_folder_path,
            ],
        tooltip="Specify the path to a BRATS dataset."
        )

    group_id = utils.forms.SelectField(
        'Group',
        choices=[
            ('HGG', 'High-Grade Group'),
            ('LGG', 'Low-Grade Group'),
            ],
        default='HGG',
        tooltip="Select a group to train on."
        )

    modality = utils.forms.SelectField(
        'Modality',
        choices=[
            ('all', 'All'),
            ('Flair', 'FLAIR'),
            ('T1', 'T1'),
            ('T1c', 'T1c'),
            ('T2', 'T2'),
            ],
        default='Flair',
        tooltip="Select a modality to train on."
        )

    filter_method = utils.forms.SelectField(
        'Filter',
        choices=[
            ('all', 'All'),
            ('max', 'Max'),
            ('threshold', 'Threshold'),
            ],
        default='all',
        tooltip="Select a slice filter: 'All' retains all axial slices, "
                "'Max' retains only the slice that exhibits max tumor area, "
                "'Threshold' retains only slices that have more than "
                "1000-pixel tumor area"
        )

    channel_conversion = utils.forms.SelectField(
        'Channel conversion',
        choices=[
            ('none', 'None - 3D grayscale images'),
            ('RGB', 'RGB - slice into 2D color images'),
            ('L', 'Grayscale - slice into 2D grayscale images'),
            ],
        default='L',
        tooltip="Perform selected channel conversion."
        )

    pct_val = utils.forms.IntegerField(
        u'% for validation',
        default=10,
        validators=[
            validators.NumberRange(min=0, max=100)
            ],
        tooltip="You can choose to set apart a certain percentage of images "
                "from the training images for the validation set."
        )


@subclass
class InferenceForm(Form):

    def validate_file_path(form, field):
        if not field.data:
            pass
        else:
            # make sure the filesystem path exists
            if not os.path.exists(field.data) and not os.path.isdir(field.data):
                raise validators.ValidationError(
                    'File does not exist or is not reachable')
            else:
                return True
    """
    A form used to perform inference on a text classification dataset
    """
    test_image_file = utils.forms.StringField(
        u'Image file',
        validators=[
            validate_file_path,
            ],
        tooltip="Provide an image"
        )

    validation_record = utils.forms.SelectField(
        'Record from validation set',
        choices=[
            ('none', '- select record -'),
            ],
        default='none',
        tooltip="Test a record from the validation set."
        )
