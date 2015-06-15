# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path
import requests

import wtforms
from wtforms import validators

from ..forms import ImageDatasetForm
from digits import utils
from digits.utils.forms import validate_required_iff

class ImageClassificationDatasetForm(ImageDatasetForm):
    """
    Defines the form used to create a new ImageClassificationDatasetJob
    """

    # Use a SelectField instead of a HiddenField so that the default value
    # is used when nothing is provided (through the REST API)
    method = wtforms.SelectField(u'Dataset type',
            choices = [
                ('folder', 'Folder'),
                ('textfile', 'Textfiles'),
                ],
            default='folder',
            )

    def validate_folder_path(form, field):
        if not field.data:
            pass
        elif utils.is_url(field.data):
            # make sure the URL exists
            try:
                r = requests.get(field.data,
                        allow_redirects=False,
                        timeout=utils.HTTP_TIMEOUT)
                if r.status_code not in [requests.codes.ok, requests.codes.moved, requests.codes.found]:
                    raise validators.ValidationError('URL not found')
            except Exception as e:
                raise validators.ValidationError('Caught %s while checking URL: %s' % (type(e).__name__, e))
            else:
                return True
        else:
            # make sure the filesystem path exists
            if not os.path.exists(field.data) or not os.path.isdir(field.data):
                raise validators.ValidationError('Folder does not exist')
            else:
                return True

    ### Method - folder

    folder_train = wtforms.StringField(u'Training Images',
            validators=[
                validate_required_iff(method='folder'),
                validate_folder_path,
                ]
            )

    folder_pct_val = wtforms.IntegerField(u'% for validation',
            default=25,
            validators=[
                validate_required_iff(method='folder'),
                validators.NumberRange(min=0, max=100)
                ]
            )

    folder_pct_test = wtforms.IntegerField(u'% for testing',
            default=0,
            validators=[
                validate_required_iff(method='folder'),
                validators.NumberRange(min=0, max=100)
                ]
            )

    has_val_folder = wtforms.BooleanField('Separate validation images folder',
            default = False,
            validators=[
                validate_required_iff(method='folder')
                ]
            )

    folder_val = wtforms.StringField(u'Validation Images',
            validators=[
                validate_required_iff(
                    method='folder',
                    has_val_folder=True),
                validate_folder_path,
                ]
            )

    has_test_folder = wtforms.BooleanField('Separate test images folder',
            default = False,
            validators=[
                validate_required_iff(method='folder')
                ]
            )

    folder_test = wtforms.StringField(u'Test Images',
            validators=[
                validate_required_iff(
                    method='folder',
                    has_test_folder=True),
                validate_folder_path,
                ]
            )

    ### Method - textfile

    textfile_train_images = wtforms.FileField(u'Training images',
            validators=[
                validate_required_iff(method='textfile')
                ]
            )
    textfile_train_folder = wtforms.StringField(u'Training images folder')

    def validate_textfile_train_folder(form, field):
        if form.method.data != 'textfile':
            field.errors[:] = []
            raise validators.StopValidation()
        if not field.data.strip():
            # allow null
            return True
        if not os.path.exists(field.data) or not os.path.isdir(field.data):
            raise validators.ValidationError('folder does not exist')
        return True

    textfile_use_val = wtforms.BooleanField(u'Validation set',
            default=True,
            validators=[
                validate_required_iff(method='textfile')
                ]
            )
    textfile_val_images = wtforms.FileField(u'Validation images',
            validators=[
                validate_required_iff(
                    method='textfile',
                    textfile_use_val=True)
                ]
            )
    textfile_val_folder = wtforms.StringField(u'Validation images folder')

    def validate_textfile_val_folder(form, field):
        if form.method.data != 'textfile' or not form.textfile_use_val.data:
            field.errors[:] = []
            raise validators.StopValidation()
        if not field.data.strip():
            # allow null
            return True
        if not os.path.exists(field.data) or not os.path.isdir(field.data):
            raise validators.ValidationError('folder does not exist')
        return True

    textfile_use_test = wtforms.BooleanField(u'Test set',
            default=False,
            validators=[
                validate_required_iff(method='textfile')
                ]
            )
    textfile_test_images = wtforms.FileField(u'Test images',
            validators=[
                validate_required_iff(
                    method='textfile',
                    textfile_use_test=True)
                ]
            )
    textfile_test_folder = wtforms.StringField(u'Test images folder')

    def validate_textfile_test_folder(form, field):
        if form.method.data != 'textfile' or not form.textfile_use_test.data:
            field.errors[:] = []
            raise validators.StopValidation()
        if not field.data.strip():
            # allow null
            return True
        if not os.path.exists(field.data) or not os.path.isdir(field.data):
            raise validators.ValidationError('folder does not exist')
        return True

    # Can't use a BooleanField here because HTML doesn't submit anything
    # for an unchecked checkbox. Since we want to use a REST API and have
    # this default to True when nothing is supplied, we have to use a
    # SelectField
    textfile_shuffle = wtforms.SelectField('Shuffle lines',
            choices = [
                (1, 'Yes'),
                (0, 'No'),
                ],
            coerce=int,
            default=1,
            )

    textfile_labels_file = wtforms.FileField(u'Labels',
            validators=[
                validate_required_iff(method='textfile')
                ]
            )

