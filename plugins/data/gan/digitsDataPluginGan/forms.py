# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

from flask.ext.wtf import Form
from wtforms import StringField, validators

from digits import utils
from digits.utils import subclass
from digits.utils.forms import validate_required_iff


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

    image_folder = utils.forms.StringField(
        u'Image folder',
        validators=[
            validators.DataRequired(),
            validate_folder_path,
            ],
        tooltip="Specify the path to a folder of images."
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
    row_count = utils.forms.IntegerField(
        u'Rows',
        default=10,
        validators=[
            validators.NumberRange(min=1)
        ],
        tooltip="Rows to generate in output grid."
    )

    task_id = utils.forms.SelectField(
        'Task ID',
        choices=[
            ('class', 'Class sweep'),
            ('style', 'Style sweep'),
            ('genimg', 'Generate single image'),
            ],
        default='class',
        tooltip="Select a task to execute."
        )

    class_z_vector = StringField(
        u'Z vector (leave blank for random)',
    )

    style_z1_vector = StringField(
        u'Z1 vector (leave blank for random)',
    )

    style_z2_vector = StringField(
        u'Z2 vector (leave blank for random)',
    )

    genimg_z_vector = StringField(
        u'Z vector (leave blank for random)',
    )

    genimg_class_id = utils.forms.IntegerField(
        u'Class ID',
        default=0,
        validators=[
            validators.NumberRange(min=0, max=9)
        ],
        tooltip="Class of image to generate."
    )