# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os

from google.protobuf import text_format
from flask.ext.wtf import Form
import wtforms
from wtforms import validators
try:
    import caffe_pb2
except ImportError:
    # See issue #32
    from caffe.proto import caffe_pb2

from digits.config import config_value
from digits.device_query import get_device, get_nvml_info
from digits.utils import sizeof_fmt
from digits.utils.forms import validate_required_iff

class ModelForm(Form):

    ### Methods

    def selection_exists_in_choices(form, field):
        found = False
        for choice in field.choices:
            if choice[0] == field.data:
                found = True
        if not found:
            raise validators.ValidationError("Selected job doesn't exist. Maybe it was deleted by another user.")

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

    ### Solver types

    solver_type = wtforms.SelectField('Solver type',
        choices = [
                ('SGD', 'Stochastic gradient descent (SGD)'),
                ('ADAGRAD', 'Adaptive gradient (AdaGrad)'),
                ('NESTEROV', "Nesterov's accelerated gradient (NAG)"),
                ],
            default = 'SGD'
            )

    ### Learning rate

    learning_rate = wtforms.FloatField('Base Learning Rate',
            default = 0.01,
            validators = [
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
                    float(value)
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

    # Use a SelectField instead of a HiddenField so that the default value
    # is used when nothing is provided (through the REST API)
    method = wtforms.SelectField(u'Network type',
            choices = [
                ('standard', 'Standard network'),
                ('previous', 'Previous network'),
                ('custom', 'Custom network'),
                ],
            default='standard',
            )

    # The options for this get set in the view (since they are dependent on the data type)
    standard_networks = wtforms.RadioField('Standard Networks',
            validators = [
                validate_required_iff(method='standard'),
                ],
            )

    previous_networks = wtforms.RadioField('Previous Networks',
            choices = [],
            validators = [
                validate_required_iff(method='previous'),
                selection_exists_in_choices,
                ],
            )

    custom_network = wtforms.TextAreaField('Custom Network',
            validators = [
                validate_required_iff(method='custom'),
                validate_NetParameter,
                ]
            )

    custom_network_snapshot = wtforms.TextField('Pretrained model')

    def validate_custom_network_snapshot(form, field):
        if form.method.data == 'custom':
            snapshot = field.data.strip()
            if snapshot:
                if not os.path.exists(snapshot):
                    raise validators.ValidationError('File does not exist')

    # Select one of several GPUs
    select_gpu = wtforms.RadioField('Select which GPU you would like to use',
            choices = [('next', 'Next available')] + [(
                index,
                '#%s - %s%s' % (
                    index,
                    get_device(index).name,
                    ' (%s memory)' % sizeof_fmt(get_nvml_info(index)['memory']['total'])
                        if get_nvml_info(index) and 'memory' in get_nvml_info(index) else '',
                    ),
                ) for index in config_value('gpu_list').split(',') if index],
            default = 'next',
            )

    # Select N of several GPUs
    select_gpus = wtforms.SelectMultipleField('Select which GPU[s] you would like to use',
            choices = [(
                index,
                '#%s - %s%s' % (
                    index,
                    get_device(index).name,
                    ' (%s memory)' % sizeof_fmt(get_nvml_info(index)['memory']['total'])
                        if get_nvml_info(index) and 'memory' in get_nvml_info(index) else '',
                    ),
                ) for index in config_value('gpu_list').split(',') if index]
            )

    def validate_select_gpus(form, field):
        # XXX For testing
        # The Flask test framework can't handle SelectMultipleFields correctly
        if hasattr(form, 'select_gpus_list'):
            field.data = form.select_gpus_list.split(',')

    # Use next available N GPUs
    select_gpu_count = wtforms.IntegerField('Use this many GPUs (next available)',
            validators = [
                validators.NumberRange(min=1, max=len(config_value('gpu_list').split(',')))
                ],
            default = 1,
            )

    def validate_select_gpu_count(form, field):
        if field.data is None:
            if form.select_gpus.data:
                # Make this field optional
                field.errors[:] = []
                raise validators.StopValidation()

    model_name = wtforms.StringField('Model Name',
            validators = [
                validators.DataRequired()
                ]
            )


