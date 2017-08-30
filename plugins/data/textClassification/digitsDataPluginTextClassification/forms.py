# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

from digits import utils
from digits.utils import subclass
from flask.ext.wtf import Form
from wtforms import validators


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


@subclass
class DatasetForm(Form):
    """
    A form used to create a text classification dataset
    """

    train_data_file = utils.forms.StringField(
        u'Training data (.csv)',
        validators=[
            validators.DataRequired(),
            validate_file_path,
        ],
        tooltip="Data file in .csv format. There should be one sample "
                "per line. On each line, the first field should be the "
                "numerical class label. All subsequent fields will be "
                "concatenated to produce a single string of characters, "
                "up to the specified limit"
    )

    val_data_file = utils.forms.StringField(
        u'Validation data (.csv)',
        validators=[
            validate_file_path,
        ],
        tooltip="Data file in .csv format. There should be one sample "
                "per line. On each line, the first field should be the "
                "numerical class label. All subsequent fields will be "
                "concatenated to produce a single string of characters, "
                "up to the specified limit."
    )

    alphabet = utils.forms.StringField(
        u'Dictionary',
        default="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}",
        tooltip="Alphabet to use when converting characters to IDs "
                "(1-based indexing). Unknown characters will be all be "
                "assigned the same next available ID. "
    )

    class_labels_file = utils.forms.StringField(
        u'Class labels',
        validators=[
            validate_file_path,
        ],
        tooltip="The 'i'th line of the file should give the string label "
                "associated with the '(i-1)'th numeric label. (E.g. the "
                "string label for the numeric label 0 is supposed to be "
                "on line 1.)"
    )

    max_chars_per_sample = utils.forms.IntegerField(
        u'Number of characters per sample',
        default=1024,
        validators=[
            validators.Optional(),
            validators.NumberRange(min=1),
        ],
        tooltip="Specify how many characters to retain per sample. "
                "Shorter samples will be padded. Longer samples will "
                "be truncated. Leave blank and disable 'Enforce same "
                "shape' below to use sample size in data file "
    )


@subclass
class InferenceForm(Form):
    """
    A form used to perform inference on a text classification dataset
    """

    snippet = utils.forms.TextAreaField(
        u'Snippet',
        tooltip="Test a single snippet"
    )

    test_data_file = utils.forms.StringField(
        u'Test data (.csv)',
        validators=[
            validate_file_path,
        ],
        tooltip="Data file in .csv format. There should be one sample "
                "per line. On each line, the first field should be the "
                "numerical class label. All subsequent fields will be "
                "concatenated to produce a single string of characters, "
                "up to the specified limit"
    )
