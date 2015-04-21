# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path
import requests

import wtforms
from wtforms import validators
from werkzeug.datastructures import FileStorage

from ..forms import ImageDatasetForm
from digits import utils

class ImageClassificationDatasetForm(ImageDatasetForm):
    """
    Defines the form used to create a new ImageClassificationDatasetJob
    """

    ### Upload method

    def required_if_method(value):

        def _required(form, field):
            if form.method.data == value:
                if field.data is None or (isinstance(field.data, str) and not field.data.strip()) or (isinstance(field.data, FileStorage) and not field.data.filename.strip()):
                    raise validators.ValidationError('This field is required.')
            else:
                field.errors[:] = []
                raise validators.StopValidation()

        return _required

    method = wtforms.HiddenField(u'Dataset type',
            default='folder',
            validators=[
                validators.AnyOf(['folder', 'textfile'], message='The method you chose is not currently supported.')
                ]
            )

    def validate_folder_path(form, field):
        if utils.is_url(field.data):
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
                required_if_method('folder'),
                validate_folder_path,
                ]
            )

    folder_pct_val = wtforms.IntegerField(u'% for validation',
            default=25,
            validators=[
                required_if_method('folder'),
                validators.NumberRange(min=0, max=100)
                ]
            )

    folder_pct_test = wtforms.IntegerField(u'% for testing',
            default=0,
            validators=[
                required_if_method('folder'),
                validators.NumberRange(min=0, max=100)
                ]
            )

    has_val_folder = wtforms.BooleanField('Separate validation images folder',
            default = False,
            validators=[
                required_if_method('folder')
                ]
            )

    folder_val = wtforms.StringField(u'Validation Images',
            validators=[
                required_if_method('folder'),
                validate_folder_path,
                ]
            )

    def validate_folder_val(form, field):
        if not form.has_val_folder.data:
            field.errors[:] = []
            raise validators.StopValidation()

    has_test_folder = wtforms.BooleanField('Separate test images folder',
            default = False,
            validators=[
                required_if_method('folder')
                ]
            )

    folder_test = wtforms.StringField(u'Test Images',
            validators=[
                required_if_method('folder'),
                validate_folder_path,
                ]
            )

    def validate_folder_test(form, field):
        if not form.has_test_folder.data:
            field.errors[:] = []
            raise validators.StopValidation()

    ### Method - textfile

    textfile_train_images = wtforms.FileField(u'Training images',
            validators=[
                required_if_method('textfile')
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


    # TODO: fix these validators

    textfile_use_val = wtforms.BooleanField(u'Validation set',
            default=True,
            validators=[
                required_if_method('textfile')
                ]
            )
    textfile_val_images = wtforms.FileField(u'Validation images',
            validators=[
                required_if_method('textfile')
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
                required_if_method('textfile')
                ]
            )
    textfile_test_images = wtforms.FileField(u'Test images',
            validators=[
                required_if_method('textfile')
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

    textfile_shuffle = wtforms.BooleanField('Shuffle lines',
            default = True)

    textfile_labels_file = wtforms.FileField(u'Labels',
            validators=[
                required_if_method('textfile')
                ]
            )

