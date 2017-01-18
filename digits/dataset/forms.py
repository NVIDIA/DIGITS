# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from flask.ext.wtf import Form
from wtforms.validators import DataRequired
from wtforms import validators

from digits import utils


class DatasetForm(Form):
    """
    Defines the form used to create a new Dataset
    (abstract class)
    """

    dataset_name = utils.forms.StringField(u'Dataset Name',
                                           validators=[DataRequired()]
                                           )

    group_name = utils.forms.StringField('Group Name',
                                         tooltip="An optional group name for organization on the main page."
                                         )

    # slurm options
    slurm_selector = utils.forms.BooleanField('Use slurm?')
    slurm_time_limit = utils.forms.IntegerField('Task time limit', tooltip='in minutes', default=0, )
    slurm_cpu_count = utils.forms.IntegerField('Use this many cores', validators=[
        validators.NumberRange(min=1, max=128)
    ], default=8, )
    slurm_mem = utils.forms.IntegerField('Use this much memory (GB)', validators=[
        validators.NumberRange(min=1, max=128)
    ], default=10, )
