# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import os.path

import wtforms
from wtforms import validators

from ..forms import ImageDatasetForm
from digits.utils.forms import validate_required_iff

class GenericImageDatasetForm(ImageDatasetForm):
    """
    Defines the form used to create a new GenericImageDatasetJob
    """

    # Use a SelectField instead of a HiddenField so that the default value
    # is used when nothing is provided (through the REST API)
    method = wtforms.SelectField(u'Dataset type',
            choices = [
                ('prebuilt', 'Prebuilt'),
                ],
            default='prebuilt',
            )

    def validate_lmdb_path(form, field):
        if not field.data:
            pass
        else:
            # make sure the filesystem path exists
            if not os.path.exists(field.data) or not os.path.isdir(field.data):
                raise validators.ValidationError('Folder does not exist')

    def validate_file_path(form, field):
        if not field.data:
            pass
        else:
            # make sure the filesystem path exists
            if not os.path.exists(field.data) or not os.path.isfile(field.data):
                raise validators.ValidationError('File does not exist')

    ### Method - prebuilt

    prebuilt_train_images = wtforms.StringField('Training Images',
            validators=[
                validate_required_iff(method='prebuilt'),
                validate_lmdb_path,
                ]
            )
    prebuilt_train_labels = wtforms.StringField('Training Labels',
            validators=[
                validate_lmdb_path,
                ]
            )
    prebuilt_val_images = wtforms.StringField('Validation Images',
            validators=[
                validate_lmdb_path,
                ]
            )
    prebuilt_val_labels = wtforms.StringField('Validation Labels',
            validators=[
                validate_lmdb_path,
                ]
            )

    prebuilt_mean_file = wtforms.StringField('Mean Image',
            validators=[
                validate_file_path,
                ]
            )

