# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import wtforms
from wtforms import validators

from ..forms import ModelForm
from digits import utils

class ImageModelForm(ModelForm):
    """
    Defines the form used to create a new ImageModelJob
    """

    crop_size = utils.forms.IntegerField('Crop Size',
            validators = [
                    validators.NumberRange(min=1),
                    validators.Optional()
                    ],
            tooltip = "If specified, during training a random square crop will be taken from the input image before using as input for the network."
            )

    # Can't use a BooleanField here because HTML doesn't submit anything
    # for an unchecked checkbox. Since we want to use a REST API and have
    # this default to True when nothing is supplied, we have to use a
    # SelectField
    use_mean = utils.forms.SelectField('Subtract Mean',
            choices = [
                ('none', 'None'),
                ('image', 'Image'),
                ('pixel', 'Pixel'),
                ],
            default='image',
            tooltip = "Subtract the mean file or mean pixel for this dataset from each image."
            )

