# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

from flask.ext.wtf import Form
from wtforms import HiddenField, validators

from digits import utils
from digits.utils import subclass
from digits.utils.forms import validate_required_iff


@subclass
class DatasetForm(Form):
    """
    A form used to create a Sunnybrook dataset
    """

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

    file_list = utils.forms.StringField(
        u'File list (with attributes) in CelebA format',
        validators=[
            validate_file_path,
        ],
        tooltip="Provide file list in CelebA format"
    )

    image_folder = utils.forms.StringField(
        u'Image folder',
        validators=[
            validators.DataRequired(),
            validate_folder_path,
            ],
        tooltip="Specify the path to a folder of images."
        )

    center_crop_size = utils.forms.IntegerField(
        u'Center crop size',
        default=108,
        validators=[
            validators.NumberRange(min=0)
        ],
        tooltip="Specify center crop."
    )

    resize = utils.forms.IntegerField(
        u'Resize after crop',
        default=64,
        tooltip="Resize after crop."
    )

@subclass
class InferenceForm(Form):
    """
    A form used to perform inference on a text classification dataset
    """

    def __init__(self, attributes, editable_attribute_ids, **kwargs):
        super(InferenceForm, self).__init__(**kwargs)
        self.attributes = attributes
        self.editable_attribute_ids = editable_attribute_ids

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

    row_count = utils.forms.IntegerField(
        u'Rows',
        default=10,
        validators=[
            validators.NumberRange(min=1)
        ],
        tooltip="Rows to generate in output grid."
    )

    dataset_type = utils.forms.SelectField(
        'Dataset',
        choices=[
            ('mnist', 'MNIST'),
            ('celeba', 'CelebA'),
            ],
        default='celeba',
        tooltip="Select a dataset."
        )

    task_id = utils.forms.SelectField(
        'Task ID',
        choices=[
            ('class', 'MNIST - Class sweep'),
            ('style', 'MNIST - Style sweep'),
            ('genimg', 'Generate single image'),
            ('attributes', 'CelebA - add/remove attributes'),
            ('enclist', 'CelebA - Encode list of images'),
            ],
        default='class',
        tooltip="Select a task to execute."
        )

    class_z_vector = utils.forms.StringField(
        u'Z vector (leave blank for random)',
    )

    style_z1_vector = utils.forms.StringField(
        u'Z1 vector (leave blank for random)',
    )

    style_z2_vector = utils.forms.StringField(
        u'Z2 vector (leave blank for random)',
    )

    genimg_z_vector = utils.forms.StringField(
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

    attributes_z_vector = utils.forms.StringField(
        u'Z vector (leave blank for random)',
    )

    attributes_file = utils.forms.StringField(
        u'Attributes vector file',
        validators=[
            validate_file_path,
            ],
        tooltip="Specify the path to a file that contains attributes vectors."
        )

    attributes_params = HiddenField()

    enc_file_list = utils.forms.StringField(
        u'File list',
        validators=[
            validate_file_path,
            ],
        tooltip="Specify the path to a file that contains a list of files."
        )

    enc_image_folder = utils.forms.StringField(
        u'Image folder',
        validators=[
            validate_folder_path,
            ],
        tooltip="Specify the path to a folder of images."
        )