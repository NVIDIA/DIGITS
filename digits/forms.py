# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

from flask.ext.wtf import Form
from wtforms import StringField
from wtforms.validators import DataRequired

class LoginForm(Form):
    username = StringField(u'Username',
        validators=[DataRequired()]
        )
