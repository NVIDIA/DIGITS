# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path
import requests

import wtforms
from wtforms import validators

from ..forms import ImageDatasetForm
from digits import utils
from digits.utils.forms import validate_required_iff

class FeatureExtractionDatasetForm(ImageDatasetForm):
    """
    Defines the form used to create a new FeatureExtractionDatasetJob
    """

    # Use a SelectField instead of a HiddenField so that the default value
    # is used when nothing is provided (through the REST API)
    method = wtforms.StringField(u'Dataset type',default='textfile')
    
    ### Method - textfile

    textfile_train_images = wtforms.FileField(u'Training images',
            validators=[
                validate_required_iff(method='textfile')
                ]
            )
    # Can't use a BooleanField here because HTML doesn't submit anything
    # for an unchecked checkbox. Since we want to use a REST API and have
    # this default to True when nothing is supplied, we have to use a
    # SelectField
    textfile_shuffle = wtforms.SelectField('Shuffle lines',
            choices = [
                (1, 'Yes'),
                (0, 'No'),
                ],
            coerce=int,
            default=1,
            )

    textfile_labels_file = wtforms.FileField(u'Labels',
            validators=[
                validate_required_iff(method='textfile')
                ]
            )
