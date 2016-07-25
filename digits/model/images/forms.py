# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

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

    # The mean_path is the file path pointing to the a protoblob of the mean image which is to be used instead of the
    #  dataset's protoblob. If it is left empty, then the protoblob from the dataset will be used instead.
    custom_mean_path = utils.forms.StringField('Custom Mean Image Protoblob File',
                                        validators = [
                                            validators.Optional()
                                            ],
                                        tooltip = "The protoblob from which to get the data used for mean subtraction. If left unspecified, the protoblob of the dataset will be used."
                                        )

