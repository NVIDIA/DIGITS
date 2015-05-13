# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os

from google.protobuf import text_format
from flask.ext.wtf import Form
import wtforms
from wtforms import validators
from caffe.proto import caffe_pb2

from digits import utils

class ModelForm(Form):

    def __init__(self, csrf_enabled=False, *args, **kwargs):
        super(ModelForm, self).__init__(csrf_enabled=csrf_enabled, *args, **kwargs)

    ### Methods
    def selection_exists_in_choices(form, field):
        found=False
        for i, choice in enumerate(field.choices):
            if choice[0] == field.data:
                found=True
        if found == False:
            raise validators.ValidationError("Selected job doesn't exist. Maybe it was deleted by another user.")

    def required_if_method(value, framework_opt = None):
        def _required(form, field):
            # second condition is to ensure that framework check will be done only when the framework parameter is provided to the required_if_method()
            if form.method.data == value and (framework_opt is None or form.framework.data == framework_opt):
                if not field.data or (isinstance(field.data, str) and field.data.strip() == ""):
                    raise validators.ValidationError('This field is required.')
            else:
                field.errors[:] = []
                raise validators.StopValidation()
        return _required

    def validate_NetParameter(form, field):
        pb = caffe_pb2.NetParameter()
        try:
            text_format.Merge(field.data, pb)
        except text_format.ParseError as e:
            raise validators.ValidationError('Not a valid NetParameter: %s' % e)

    ### Fields

    # The options for this get set in the view (since they are dynamic)
    dataset = wtforms.SelectField('Select Dataset', choices=[])

    train_epochs = wtforms.IntegerField('Training epochs',
            validators = [
                validators.NumberRange(min=1)
                ],
            default=30,
            )

    snapshot_interval = wtforms.FloatField('Snapshot interval (in epochs)',
            default = 1,
            validators = [
                validators.NumberRange(min=0),
                ],
            )

    val_interval = wtforms.FloatField('Validation interval (in epochs)',
            default = 1,
            validators = [
                validators.NumberRange(min=0)
                ],
            )

    random_seed = wtforms.IntegerField('Random seed',
            validators = [
                validators.NumberRange(min=0),
                validators.Optional(),
                ],
            )

    batch_size = wtforms.IntegerField('Batch size',
            validators = [
                validators.NumberRange(min=1),
                validators.Optional(),
                ],
            )

    ### Learning rate

    learning_rate = wtforms.FloatField('Base Learning Rate',
            default = 0.01,
            validators = [
                validators.DataRequired(),
                validators.NumberRange(min=0),
                ]
            )

    lr_policy = wtforms.SelectField('Policy',
            choices = [
                ('fixed', 'Fixed'),
                ('step', 'Step Down'),
                ('multistep', 'Step Down (arbitrary steps)'),
                ('exp', 'Exponential Decay'),
                ('inv', 'Inverse Decay'),
                ('poly', 'Polynomial Decay'),
                ('sigmoid', 'Sigmoid Decay'),
                ],
            default = 'step'
            )

    lr_step_size = wtforms.FloatField('Step Size',
            default = 33
            )
    lr_step_gamma = wtforms.FloatField('Gamma',
            default = 0.1
            )
    lr_multistep_values = wtforms.StringField('Step Values',
            default = "50,85"
            )
    def validate_lr_multistep_values(form, field):
        if form.lr_policy.data == 'multistep':
            for value in field.data.split(','):
                try:
                    v = float(value)
                except ValueError:
                    raise validators.ValidationError('invalid value')
    lr_multistep_gamma = wtforms.FloatField('Gamma',
            default = 0.5
            )
    lr_exp_gamma = wtforms.FloatField('Gamma',
            default = 0.95
            )
    lr_inv_gamma = wtforms.FloatField('Gamma',
            default = 0.1
            )
    lr_inv_power = wtforms.FloatField('Power',
            default = 0.5
            )
    lr_poly_power = wtforms.FloatField('Power',
            default = 3
            )
    lr_sigmoid_step = wtforms.FloatField('Step',
            default = 50
            )
    lr_sigmoid_gamma = wtforms.FloatField('Gamma',
            default = 0.1
            )

    ### Network

    method = wtforms.HiddenField('Model type',
            validators = [
                validators.AnyOf(
                    ['standard', 'previous', 'custom'],
                    message='The method you chose is not currently supported.'
                    )
                ],
            default = 'standard',
            )

    ## framework
    framework = wtforms.HiddenField('framework',
            validators = [
                validators.AnyOf(
                    ['caffe', 'torch'],
                    message='The framework you choose is not currently supported.'
                    )
                ],
            default = 'caffe'
            )

    # The options for this get set in the view (since they are dependent on the data type)
    standard_networks = wtforms.RadioField('Standard Networks',
            validators = [
                required_if_method('standard'),
                ],
            )

    previous_networks = wtforms.RadioField('Previous Networks',
            choices = [],
            validators = [
                required_if_method('previous'),
                selection_exists_in_choices,
                ],
            )

    # custom network validation is required for the caffe framework, because of limited protobuf support for torch framework.
    caffe_custom_network = wtforms.TextAreaField('Caffe Custom Network',
            validators = [
                required_if_method('custom','caffe'),
                validate_NetParameter,
                ]
            )

    torch_custom_network = wtforms.TextAreaField('Torch Custom Network',
            validators = [
                required_if_method('custom','torch')
                ]
            )

    custom_network_snapshot = wtforms.TextField('Pretrained model')

    def validate_custom_network_snapshot(form, field):
        if form.method.data == 'custom':
            snapshot = field.data.strip()
            if snapshot:
                if not os.path.exists(snapshot):
                    raise validators.ValidationError('File does not exist')

    model_name = wtforms.StringField('Model Name',
            validators = [
                validators.DataRequired()
                ]
            )

    shuffle = wtforms.BooleanField('Shuffle Train Data',
                        default = True
            )

