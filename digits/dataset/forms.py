# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

from flask.ext.wtf import Form
from wtforms import StringField
from wtforms.validators import DataRequired

class DatasetForm(Form):
    """
    Defines the form used to create a new Dataset
    (abstract class)
    """

    dataset_name = StringField(u'Dataset Name',
            validators=[DataRequired()]
            )

