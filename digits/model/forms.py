# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os

from flask.ext.wtf import Form
import wtforms
from wtforms import validators

from digits.config import config_value
from digits.device_query import get_device, get_nvml_info
from digits import utils
from digits.utils import sizeof_fmt
from digits.utils.forms import validate_required_iff
from digits import frameworks

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
        fw = frameworks.get_framework_by_id(form['framework'].data)
        try:
            fw.validate_network(field.data)
        except frameworks.exceptions.BadNetworkException as e:
            raise validators.ValidationError('Bad network: %s' % e.message)

    ### Fields

    # The options for this get set in the view (since they are dynamic)
    dataset = utils.forms.SelectField('Select Dataset', choices=[],
                tooltip = "Choose the dataset to use for this model."
            )

    train_epochs = utils.forms.IntegerField('Training epochs',
            validators = [
                validators.NumberRange(min=1)
                ],
            default=30,
            tooltip = "How many passes through the training data?"
            )

    snapshot_interval = utils.forms.FloatField('Snapshot interval (in epochs)',
            default = 1,
            validators = [
                validators.NumberRange(min=0),
                ],
            tooltip = "How many epochs of training between taking a snapshot?"
            )

    val_interval = utils.forms.FloatField('Validation interval (in epochs)',
            default = 1,
            validators = [
                validators.NumberRange(min=0)
                ],
            tooltip = "How many epochs of training between running through one pass of the validation data?"
            )

    random_seed = utils.forms.IntegerField('Random seed',
            validators = [
                validators.NumberRange(min=0),
                validators.Optional(),
                ],
            tooltip = "If you provide a random seed, then back-to-back runs with the same model and dataset should give identical results."
            )

    batch_size = utils.forms.IntegerField('Batch size',
            validators = [
                validators.NumberRange(min=1),
                validators.Optional(),
                ],
            tooltip = "How many images to process at once. If blank, values are used from the network definition."
            )

    ### Solver types

    solver_type = utils.forms.SelectField('Solver type',
        choices = [
                ('SGD', 'Stochastic gradient descent (SGD)'),
                ('ADAGRAD', 'Adaptive gradient (AdaGrad)'),
                ('NESTEROV', "Nesterov's accelerated gradient (NAG)"),
                ],
            default = 'SGD',
            tooltip = "What type of solver will be used??"
            )

    ### Learning rate

    learning_rate = utils.forms.FloatField('Base Learning Rate',
            default = 0.01,
            validators = [
                validators.NumberRange(min=0),
                ],
            tooltip = "Affects how quickly the network learns. If you are getting NaN for your loss, you probably need to lower this value."
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

    ## framework
    framework = wtforms.HiddenField('framework',
            validators = [
                validators.AnyOf(
                    [fw.get_id() for fw in frameworks.get_frameworks()],
                    message='The framework you choose is not currently supported.'
                    )
                ],
            default = frameworks.get_frameworks()[0].get_id()
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

    custom_network = utils.forms.TextAreaField('Custom Network',
            validators = [
                validate_required_iff(method='custom'),
                validate_NetParameter,
                ],
            )

    custom_network_snapshot = utils.forms.TextField('Pretrained model',
                tooltip = "Path to pretrained model file. Only edit this field if you understand how fine-tuning works in caffe"
            )


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
    select_gpus = utils.forms.SelectMultipleField('Select which GPU[s] you would like to use',
            choices = [(
                index,
                '#%s - %s%s' % (
                    index,
                    get_device(index).name,
                    ' (%s memory)' % sizeof_fmt(get_nvml_info(index)['memory']['total'])
                        if get_nvml_info(index) and 'memory' in get_nvml_info(index) else '',
                    ),
                ) for index in config_value('gpu_list').split(',') if index],
            tooltip = "The job won't start until all of the chosen GPUs are available."
            )

    # XXX For testing
    # The Flask test framework can't handle SelectMultipleFields correctly
    select_gpus_list = wtforms.StringField('Select which GPU[s] you would like to use (comma separated)')

    def validate_select_gpus(form, field):
        if form.select_gpus_list.data:
            field.data = form.select_gpus_list.data.split(',')

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

    model_name = utils.forms.StringField('Model Name',
            validators = [
                validators.DataRequired()
                ],
            tooltip = "An identifier, later used to refer to this model in the Application."
            )

    shuffle = utils.forms.BooleanField('Shuffle Train Data',
                                       default = True,
                                       tooltip = 'For every epoch, shuffle the data before training.'
            )

