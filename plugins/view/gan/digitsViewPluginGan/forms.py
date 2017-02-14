# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from digits import utils
from digits.utils import subclass
from flask.ext.wtf import Form


@subclass
class ConfigForm(Form):
    """
    A form used to configure gradient visualization
    """

    gan_view_task_id = utils.forms.SelectField(
        'Task',
        choices=[
            ('grid', 'Grid'),
            ('mnist_encoder', 'MNIST Encoder'),
            ('celeba_encoder', 'CelebA Encoder'),
            ],
        default='grid',
        tooltip="Select a task."
        )

    pass
