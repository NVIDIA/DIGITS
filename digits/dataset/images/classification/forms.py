# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

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
                                  choices=[
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

    compression = utils.forms.SelectField(
        'DB compression',
        choices=[
            ('none', 'None'),
            ('gzip', 'GZIP'),
        ],
        default='none',
        tooltip=('Compressing the dataset may significantly decrease the size '
                 'of your database files, but it may increase read and write times.'),
    )

    # Use a SelectField instead of a HiddenField so that the default value
    # is used when nothing is provided (through the REST API)
    method = wtforms.SelectField(u'Dataset type',
                                 choices=[
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
            # and make sure the filesystem path is absolute
            if not os.path.exists(field.data) or not os.path.isdir(field.data):
                raise validators.ValidationError('Folder does not exist')
            elif not os.path.isabs(field.data):
                raise validators.ValidationError('Filesystem path is not absolute')
            else:
                return True

    #
    # Method - folder
    #

    folder_train = utils.forms.StringField(
        u'Training Images',
        validators=[
            validate_required_iff(method='folder'),
            validate_folder_path,
        ],
        tooltip=('Indicate a folder which holds subfolders full of images. '
                 'Each subfolder should be named according to the desired label for the images that it holds. '
                 'Can also be a URL for an apache/nginx auto-indexed folder.'),
    )

    folder_pct_val = utils.forms.IntegerField(
        u'% for validation',
        default=25,
        validators=[
            validate_required_iff(method='folder'),
            validators.NumberRange(min=0, max=100)
        ],
        tooltip=('You can choose to set apart a certain percentage of images '
                 'from the training images for the validation set.'),
    )

    folder_pct_test = utils.forms.IntegerField(
        u'% for testing',
        default=0,
        validators=[
            validate_required_iff(method='folder'),
            validators.NumberRange(min=0, max=100)
        ],
        tooltip=('You can choose to set apart a certain percentage of images '
                 'from the training images for the test set.'),
    )

    folder_train_min_per_class = utils.forms.IntegerField(
        u'Minimum samples per class',
        default=2,
        validators=[
            validators.Optional(),
            validators.NumberRange(min=1),
        ],
        tooltip=('You can choose to specify a minimum number of samples per class. '
                 'If a class has fewer samples than the specified amount it will be ignored. '
                 'Leave blank to ignore this feature.'),
    )

    folder_train_max_per_class = utils.forms.IntegerField(
        u'Maximum samples per class',
        validators=[
            validators.Optional(),
            validators.NumberRange(min=1),
            validate_greater_than('folder_train_min_per_class'),
        ],
        tooltip=('You can choose to specify a maximum number of samples per class. '
                 'If a class has more samples than the specified amount extra samples will be ignored. '
                 'Leave blank to ignore this feature.'),
    )

    has_val_folder = wtforms.BooleanField(
        'Separate validation images folder',
        default=False,
        validators=[
            validate_required_iff(method='folder')
        ]
    )

    folder_val = wtforms.StringField(
        u'Validation Images',
        validators=[
            validate_required_iff(
                method='folder',
                has_val_folder=True),
        ]
    )

    folder_val_min_per_class = utils.forms.IntegerField(
        u'Minimum samples per class',
        default=2,
        validators=[
            validators.Optional(),
            validators.NumberRange(min=1),
        ],
        tooltip=('You can choose to specify a minimum number of samples per class. '
                 'If a class has fewer samples than the specified amount it will be ignored. '
                 'Leave blank to ignore this feature.'),
    )

    folder_val_max_per_class = utils.forms.IntegerField(
        u'Maximum samples per class',
        validators=[
            validators.Optional(),
            validators.NumberRange(min=1),
            validate_greater_than('folder_val_min_per_class'),
        ],
        tooltip=('You can choose to specify a maximum number of samples per class. '
                 'If a class has more samples than the specified amount extra samples will be ignored. '
                 'Leave blank to ignore this feature.'),
    )

    has_test_folder = wtforms.BooleanField(
        'Separate test images folder',
        default=False,
        validators=[
            validate_required_iff(method='folder')
        ]
    )

    folder_test = wtforms.StringField(
        u'Test Images',
        validators=[
            validate_required_iff(
                method='folder',
                has_test_folder=True),
            validate_folder_path,
        ]
    )

    folder_test_min_per_class = utils.forms.IntegerField(
        u'Minimum samples per class',
        default=2,
        validators=[
            validators.Optional(),
            validators.NumberRange(min=1)
        ],
        tooltip=('You can choose to specify a minimum number of samples per class. '
                 'If a class has fewer samples than the specified amount it will be ignored. '
                 'Leave blank to ignore this feature.'),
    )

    folder_test_max_per_class = utils.forms.IntegerField(
        u'Maximum samples per class',
        validators=[
            validators.Optional(),
            validators.NumberRange(min=1),
            validate_greater_than('folder_test_min_per_class'),
        ],
        tooltip=('You can choose to specify a maximum number of samples per class. '
                 'If a class has more samples than the specified amount extra samples will be ignored. '
                 'Leave blank to ignore this feature.'),
    )

    #
    # Method - textfile
    #

    textfile_use_local_files = wtforms.BooleanField(
        u'Use local files',
        default=False,
    )

    textfile_train_images = utils.forms.FileField(
        u'Training images',
        validators=[
            validate_required_iff(method='textfile',
                                  textfile_use_local_files=False)
        ]
    )

    textfile_local_train_images = wtforms.StringField(
        u'Training images',
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
    textfile_shuffle = utils.forms.SelectField(
        'Shuffle lines',
        choices=[
            (1, 'Yes'),
            (0, 'No'),
        ],
        coerce=int,
        default=1,
        tooltip="Shuffle the list[s] of images before creating the database."
    )

    textfile_labels_file = utils.forms.FileField(
        u'Labels',
        validators=[
            validate_required_iff(method='textfile',
                                  textfile_use_local_files=False)
        ],
        tooltip=("The 'i'th line of the file should give the string label "
                 "associated with the '(i-1)'th numeric label. (E.g. the string label "
                 "for the numeric label 0 is supposed to be on line 1.)"),
    )

    textfile_local_labels_file = utils.forms.StringField(
        u'Labels',
        validators=[
            validate_required_iff(method='textfile',
                                  textfile_use_local_files=True)
        ],
        tooltip=("The 'i'th line of the file should give the string label "
                 "associated with the '(i-1)'th numeric label. (E.g. the string label "
                 "for the numeric label 0 is supposed to be on line 1.)"),
    )
