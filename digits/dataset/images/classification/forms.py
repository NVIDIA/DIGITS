# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path

import requests
from wtforms import StringField, SelectField, IntegerField, HiddenField, FileField, RadioField, TextAreaField, FormField, BooleanField
from wtforms.validators import ValidationError, StopValidation, Optional, DataRequired, NumberRange, AnyOf
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
                    raise ValidationError('This field is required.')
            else:
                field.errors[:] = []
                raise StopValidation()

        return _required

    method = HiddenField(u'Dataset type',
            default='folder',
            validators=[
                AnyOf(['folder', 'textfile'], message='The method you chose is not currently supported.')
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
                    raise ValidationError('URL not found')
            except Exception as e:
                raise ValidationError('Caught %s while checking URL: %s' % (type(e).__name__, e))
            else:
                return True
        else:
            # make sure the filesystem path exists
            if not os.path.exists(field.data) or not os.path.isdir(field.data):
                raise ValidationError('Folder does not exist')
            else:
                return True

    ### Method - folder

    folder_train = StringField(u'Training Images',
            validators=[
                required_if_method('folder'),
                validate_folder_path,
                ]
            )

    folder_pct_val = IntegerField(u'% for validation',
            default=25,
            validators=[
                required_if_method('folder'),
                NumberRange(min=0, max=100)
                ]
            )

    folder_pct_test = IntegerField(u'% for testing',
            default=0,
            validators=[
                required_if_method('folder'),
                NumberRange(min=0, max=100)
                ]
            )

    has_val_folder = BooleanField('Separate validation images folder',
            default = False,
            validators=[
                required_if_method('folder')
                ]
            )

    folder_val = StringField(u'Validation Images',
            validators=[
                required_if_method('folder'),
                validate_folder_path,
                ]
            )

    def validate_folder_val(form, field):
        if not form.has_val_folder.data:
            field.errors[:] = []
            raise StopValidation()

    has_test_folder = BooleanField('Separate test images folder',
            default = False,
            validators=[
                required_if_method('folder')
                ]
            )

    folder_test = StringField(u'Test Images',
            validators=[
                required_if_method('folder'),
                validate_folder_path,
                ]
            )

    def validate_folder_test(form, field):
        if not form.has_test_folder.data:
            field.errors[:] = []
            raise StopValidation()

    ### Method - textfile

    textfile_train_images = FileField(u'Training images',
            validators=[
                required_if_method('textfile')
                ]
            )
    textfile_train_folder = StringField(u'Training images folder')

    def validate_textfile_train_folder(form, field):
        if form.method.data != 'textfile':
            field.errors[:] = []
            raise StopValidation()
        if not field.data.strip():
            # allow null
            return True
        if not os.path.exists(field.data) or not os.path.isdir(field.data):
            raise ValidationError('folder does not exist')
        return True


    # TODO: fix these validators

    textfile_use_val = BooleanField(u'Validation set',
            default=True,
            validators=[
                required_if_method('textfile')
                ]
            )
    textfile_val_images = FileField(u'Validation images',
            validators=[
                required_if_method('textfile')
                ]
            )
    textfile_val_folder = StringField(u'Validation images folder')

    def validate_textfile_val_folder(form, field):
        if form.method.data != 'textfile' or not form.textfile_use_val.data:
            field.errors[:] = []
            raise StopValidation()
        if not field.data.strip():
            # allow null
            return True
        if not os.path.exists(field.data) or not os.path.isdir(field.data):
            raise ValidationError('folder does not exist')
        return True

    textfile_use_test = BooleanField(u'Test set',
            default=False,
            validators=[
                required_if_method('textfile')
                ]
            )
    textfile_test_images = FileField(u'Test images',
            validators=[
                required_if_method('textfile')
                ]
            )
    textfile_test_folder = StringField(u'Test images folder')

    def validate_textfile_test_folder(form, field):
        if form.method.data != 'textfile' or not form.textfile_use_test.data:
            field.errors[:] = []
            raise StopValidation()
        if not field.data.strip():
            # allow null
            return True
        if not os.path.exists(field.data) or not os.path.isdir(field.data):
            raise ValidationError('folder does not exist')
        return True

    textfile_labels_file = FileField(u'Labels',
            validators=[
                required_if_method('textfile')
                ]
            )

