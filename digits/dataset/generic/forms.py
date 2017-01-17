# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import wtforms
from wtforms import validators

from ..forms import DatasetForm
from digits import utils


class GenericDatasetForm(DatasetForm):
    """
    Defines the form used to create a new GenericDatasetJob
    """
    # Generic dataset options
    dsopts_feature_encoding = utils.forms.SelectField(
        'Feature Encoding',
        default='png',
        choices=[('none', 'None'),
                 ('png', 'PNG (lossless)'),
                 ('jpg', 'JPEG (lossy, 90% quality)'),
                 ],
        tooltip="Using either of these compression formats can save disk"
                " space, but can also require marginally more time for"
                " training."
    )

    dsopts_label_encoding = utils.forms.SelectField(
        'Label Encoding',
        default='none',
        choices=[
            ('none', 'None'),
            ('png', 'PNG (lossless)'),
            ('jpg', 'JPEG (lossy, 90% quality)'),
        ],
        tooltip="Using either of these compression formats can save disk"
                " space, but can also require marginally more time for"
                " training."
    )

    dsopts_batch_size = utils.forms.IntegerField(
        'Encoder batch size',
        validators=[
            validators.DataRequired(),
            validators.NumberRange(min=1),
        ],
        default=32,
        tooltip="Encode data in batches of specified number of entries"
    )

    dsopts_num_threads = utils.forms.IntegerField(
        'Number of encoder threads',
        validators=[
            validators.DataRequired(),
            validators.NumberRange(min=1),
        ],
        default=4,
        tooltip="Use specified number of encoder threads"
    )

    dsopts_backend = wtforms.SelectField(
        'DB backend',
        choices=[
            ('lmdb', 'LMDB'),
        ],
        default='lmdb',
    )

    dsopts_force_same_shape = utils.forms.SelectField(
        'Enforce same shape',
        choices=[
            (1, 'Yes'),
            (0, 'No'),
        ],
        coerce=int,
        default=1,
        tooltip="Check that each entry in the database has the same shape."
        "Disabling this will also disable mean image computation."
    )
