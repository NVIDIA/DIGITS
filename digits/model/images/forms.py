# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import wtforms
from wtforms import validators

from ..forms import ModelForm

class ImageModelForm(ModelForm):
    """
    Defines the form used to create a new ImageModelJob
    """

    crop_size = wtforms.IntegerField('Crop Size',
            validators = [
                    validators.NumberRange(min=1),
                    validators.Optional()
                    ]
            )

    # Can't use a BooleanField here because HTML doesn't submit anything
    # for an unchecked checkbox. Since we want to use a REST API and have
    # this default to True when nothing is supplied, we have to use a
    # SelectField
    use_mean = wtforms.SelectField('Subtract Mean File',
            choices = [
                (1, 'Yes'),
                (0, 'No'),
                ],
            coerce=int,
            default=1,
            )

