# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

from digits import utils
from digits.utils import subclass
from flask_wtf import Form
import wtforms.validators


@subclass
class ConfigForm(Form):
    """
    A form used to configure gradient visualization
    """

    def validate_file_path(form, field):
        if not field.data:
            pass
        else:
            # make sure the filesystem path exists
            if not os.path.exists(field.data) and not os.path.isdir(field.data):
                raise wtforms.validators.ValidationError('File does not exist or is not reachable')
            else:
                return True

    gan_view_task_id = utils.forms.SelectField(
        'Task',
        choices=[
            ('grid', 'Grid'),
            ('mnist_encoder', 'MNIST Encoder'),
            ('celeba_encoder', 'CelebA Encoder'),
            ('animation', 'Animation'),
            ('attributes', 'CelebA get attributes'),
            ],
        default='grid',
        tooltip="Select a task."
        )

    attributes_file = utils.forms.StringField(
        u'Attributes vector file',
        validators=[
            validate_file_path,
            ],
        tooltip="Specify the path to a file that contains attributes vectors."
        )

    pass
