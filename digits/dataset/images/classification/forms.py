# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path
import requests

import wtforms
from wtforms import validators

from ..forms import ImageDatasetForm
from digits import utils
from digits.utils.forms import validate_required_iff, validate_greater_than

class ImageClassificationDatasetForm(ImageDatasetForm):
    """
    Defines the form used to create a new ImageClassificationDatasetJob
    """

    backend = wtforms.SelectField('DB backend',
            choices = [
                ('lmdb', 'LMDB'),
                ('hdf5', 'HDF5'),
                ],
            default='lmdb',
            )

    def validate_backend(form, field):
        if field.data == 'lmdb':
            form.compression.data = 'none'
        elif field.data == 'hdf5':
            form.encoding.data = 'none'

    compression = utils.forms.SelectField('DB compression',
            choices = [
                ('none', 'None'),
                ('gzip', 'GZIP'),
                ],
            default='none',
            tooltip='Compressing the dataset may significantly decrease the size of your database files, but it may increase read and write times.',
            )

    ### Data augmentation - rotation 
    has_augmentation_rotation = utils.forms.BooleanField('Perform rotations on images',
            default = False,
            tooltip = "You can augment your dataset by performing rotation on images.",
            validators=[ 
                ]
            )

    augmentation_rotation_angle_min = utils.forms.IntegerField(u'angle min',
            default=-90,
            validators=[ 
                validators.NumberRange(min=-360, max=360)
                ],
            tooltip = "The minimum rotation angle that can be performed during augmentation."
            )

    augmentation_rotation_angle_max = utils.forms.IntegerField(u'angle max',
            default=90,
            validators=[ 
                validators.NumberRange(min=-360, max=360)
                ],
            tooltip = "The maximum rotation angle that can be performed during augmentation."
            )

    augmentation_rotation_probability = utils.forms.FloatField(u'probability',
            default=0.75,
            validators=[ 
                validators.NumberRange(min=0.0, max=1.0)
                ],
            tooltip = "The probability for an image to be rotated during augmentation."
            )

    ### Data augmentation - hue 
    has_augmentation_hue = utils.forms.BooleanField('Perform hue modulation on images',
            tooltip="You can augment your dataset by performing hue modulation on images.",
            default=False,
            validators=[ 
                ]
            )
 
    augmentation_hue_angle = utils.forms.FloatField(u'angle',
            default=0.5,
            validators=[ 
                validators.NumberRange(min=0.0, max=1.0)
                ],
            tooltip = "The maximum angle of the hue modulation."
            )

    augmentation_hue_angle_min = utils.forms.IntegerField(u'angle min',
            default=66,
            validators=[ 
                validators.NumberRange(min=0, max=360)
                ],
            tooltip = "The minimum angle of the hue modulation."
            )
    augmentation_hue_angle_max = utils.forms.IntegerField(u'angle max',
            default=122,
            validators=[ 
                validators.NumberRange(min=0, max=360)
                ],
            tooltip = "The maximum angle of the hue modulation."
            )

    augmentation_hue_probability = utils.forms.FloatField(u'probability',
            default=0.75,
            validators=[ 
                validators.NumberRange(min=0.0, max=1.0)
                ],
            tooltip = "The probability for an image to be hue-modulated during augmentation."
            )

    ### Data augmentation - contrast
    has_augmentation_contrast = utils.forms.BooleanField('Perform contrast modulation on images',
            tooltip="You can augment your dataset by performing contrast modulation on images.",
            default=False,
            validators=[ 
                ]
            )
 
    augmentation_contrast_strength_min = utils.forms.FloatField(u'strength min',
            default=0.75,
            validators=[ 
                validators.NumberRange(min=0.0, max=2.0)
                ],
            tooltip = "The minimum strength of the contrast modulation."
            )

    augmentation_contrast_strength_max = utils.forms.FloatField(u'strength max',
            default=1.25,
            validators=[ 
                validators.NumberRange(min=0.0, max=2.0)
                ],
            tooltip = "The maximum strength of the contrast modulation."
            )

    augmentation_contrast_probability = utils.forms.FloatField(u'probability',
            default=0.75,
            validators=[ 
                validators.NumberRange(min=0.0, max=1.0)
                ],
            tooltip = "The probability for an image to be contrast-modulated during augmentation."
            )

    ### Data augmentation - translation
    has_augmentation_translation = utils.forms.BooleanField('Perform translation on images',
            tooltip="You can augment your dataset by performing translation on images.",
            default=False,
            validators=[ 
                ]
            )
 
    augmentation_translation_dx_min = utils.forms.FloatField(u'dx min',
            default=-0.5,
            validators=[ 
                validators.NumberRange(min=-1.0, max=1.0)
                ],
            tooltip = "The minimum dx of the translation."
            )

    augmentation_translation_dx_max = utils.forms.FloatField(u'dx max',
            default=0.5,
            validators=[ 
                validators.NumberRange(min=-1.0, max=1.0)
                ],
            tooltip = "The maximum dx of the translation."
            )

    augmentation_translation_dy_min = utils.forms.FloatField(u'dy min',
            default=-0.5,
            validators=[ 
                validators.NumberRange(min=-1.0, max=1.0)
                ],
            tooltip = "The minimum dy of the translation."
            )

    augmentation_translation_dy_max = utils.forms.FloatField(u'dy max',
            default=0.5,
            validators=[ 
                validators.NumberRange(min=-1.0, max=1.0)
                ],
            tooltip = "The maximum dy of the translation."
            )

    augmentation_translation_probability = utils.forms.FloatField(u'probability',
            default=0.75,
            validators=[ 
                validators.NumberRange(min=0.0, max=1.0)
                ],
            tooltip = "The probability for an image to be translation-modulated during augmentation."
            )


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

    folder_train = utils.forms.StringField(u'Training Images',
            validators=[
                validate_required_iff(method='folder'),
                validate_folder_path,
                ],
            tooltip = "Indicate a folder which holds subfolders full of images. Each subfolder should be named according to the desired label for the images that it holds. Can also be a URL for an apache/nginx auto-indexed folder."
            )

    folder_pct_val = utils.forms.IntegerField(u'% for validation',
            default=25,
            validators=[
                validate_required_iff(method='folder'),
                validators.NumberRange(min=0, max=100)
                ],
            tooltip = "You can choose to set apart a certain percentage of images from the training images for the validation set."
            )

    folder_pct_test = utils.forms.IntegerField(u'% for testing',
            default=0,
            validators=[
                validate_required_iff(method='folder'),
                validators.NumberRange(min=0, max=100)
                ],
            tooltip = "You can choose to set apart a certain percentage of images from the training images for the test set."
            )

    folder_train_min_per_class = utils.forms.IntegerField(u'Minimum samples per class',
            default=2,
            validators=[
                validators.Optional(),
                validators.NumberRange(min=1),
                ],
            tooltip = "You can choose to specify a minimum number of samples per class. If a class has fewer samples than the specified amount it will be ignored. Leave blank to ignore this feature."
            )

    folder_train_max_per_class = utils.forms.IntegerField(u'Maximum samples per class',
            validators=[
                validators.Optional(),
                validators.NumberRange(min=1),
                validate_greater_than('folder_train_min_per_class'),
                ],
            tooltip = "You can choose to specify a maximum number of samples per class. If a class has more samples than the specified amount extra samples will be ignored. Leave blank to ignore this feature."
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
                ]
            )

    folder_val_min_per_class = utils.forms.IntegerField(u'Minimum samples per class',
            default=2,
            validators=[
                validators.Optional(),
                validators.NumberRange(min=1),
                ],
            tooltip = "You can choose to specify a minimum number of samples per class. If a class has fewer samples than the specified amount it will be ignored. Leave blank to ignore this feature."
            )

    folder_val_max_per_class = utils.forms.IntegerField(u'Maximum samples per class',
            validators=[
                validators.Optional(),
                validators.NumberRange(min=1),
                validate_greater_than('folder_val_min_per_class'),
                ],
            tooltip = "You can choose to specify a maximum number of samples per class. If a class has more samples than the specified amount extra samples will be ignored. Leave blank to ignore this feature."
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

    folder_test_min_per_class = utils.forms.IntegerField(u'Minimum samples per class',
            default=2,
            validators=[
                validators.Optional(),
                validators.NumberRange(min=1)
                ],
            tooltip = "You can choose to specify a minimum number of samples per class. If a class has fewer samples than the specified amount it will be ignored. Leave blank to ignore this feature."
            )

    folder_test_max_per_class = utils.forms.IntegerField(u'Maximum samples per class',
            validators=[
                validators.Optional(),
                validators.NumberRange(min=1),
                validate_greater_than('folder_test_min_per_class'),
                ],
            tooltip = "You can choose to specify a maximum number of samples per class. If a class has more samples than the specified amount extra samples will be ignored. Leave blank to ignore this feature."
            )

    ### Method - textfile

    textfile_use_local_files = wtforms.BooleanField(u'Use local files',
        default=False)

    textfile_train_images = utils.forms.FileField(u'Training images',
            validators=[
                validate_required_iff(method='textfile',
                                      textfile_use_local_files=False)
                ]
            )
    textfile_local_train_images = wtforms.StringField(u'Training images',
            validators=[
                validate_required_iff(method='textfile',
                                      textfile_use_local_files=True)
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
    textfile_val_images = utils.forms.FileField(u'Validation images',
            validators=[
                validate_required_iff(
                    method='textfile',
                    textfile_use_val=True,
                    textfile_use_local_files=False)
                ]
            )
    textfile_local_val_images = wtforms.StringField(u'Validation images',
            validators=[
                validate_required_iff(
                    method='textfile',
                    textfile_use_val=True,
                    textfile_use_local_files=True)
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
    textfile_test_images = utils.forms.FileField(u'Test images',
            validators=[
                validate_required_iff(
                    method='textfile',
                    textfile_use_test=True,
                    textfile_use_local_files=False)
                ]
            )
    textfile_local_test_images = wtforms.StringField(u'Test images',
            validators=[
                validate_required_iff(
                    method='textfile',
                    textfile_use_test=True,
                    textfile_use_local_files=True)
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
    textfile_shuffle = utils.forms.SelectField('Shuffle lines',
            choices = [
                (1, 'Yes'),
                (0, 'No'),
                ],
            coerce=int,
            default=1,
            tooltip = "Shuffle the list[s] of images before creating the database."
            )

    textfile_labels_file = utils.forms.FileField(u'Labels',
            validators=[
                validate_required_iff(method='textfile',
                                      textfile_use_local_files=False)
                ],
            tooltip = "The 'i'th line of the file should give the string label associated with the '(i-1)'th numberic label. (E.g. the string label for the numeric label 0 is supposed to be on line 1.)"
            )

    textfile_local_labels_file = utils.forms.StringField(u'Labels',
            validators=[
                validate_required_iff(method='textfile',
                                      textfile_use_local_files=True)
                ],
            tooltip = "The 'i'th line of the file should give the string label associated with the '(i-1)'th numberic label. (E.g. the string label for the numeric label 0 is supposed to be on line 1.)"
            )

