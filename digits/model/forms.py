# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.

import os

import flask
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

    # Methods

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
            # below function raises a BadNetworkException in case of validation error
            fw.validate_network(field.data)
        except frameworks.errors.BadNetworkError as e:
            raise validators.ValidationError('Bad network: %s' % e.message)

    def validate_file_exists(form, field):
        from_client = bool(form.python_layer_from_client.data)

        filename = ''
        if not from_client and field.type == 'StringField':
            filename = field.data

        if filename == '':
            return

        if not os.path.isfile(filename):
            raise validators.ValidationError('Server side file, %s, does not exist.' % filename)

    def validate_py_ext(form, field):
        from_client = bool(form.python_layer_from_client.data)

        filename = ''
        if from_client and field.type == 'FileField':
            filename = flask.request.files[field.name].filename
        elif not from_client and field.type == 'StringField':
            filename = field.data

        if filename == '':
            return

        (root, ext) = os.path.splitext(filename)
        if ext != '.py' and ext != '.pyc':
            raise validators.ValidationError('Python file, %s, needs .py or .pyc extension.' % filename)

    # Fields

    # The options for this get set in the view (since they are dynamic)
    dataset = utils.forms.SelectField(
        'Select Dataset',
        choices=[],
        tooltip="Choose the dataset to use for this model."
    )

    python_layer_from_client = utils.forms.BooleanField(
        u'Use client-side file',
        default=False,
    )

    python_layer_client_file = utils.forms.FileField(
        u'Client-side file',
        validators=[
            validate_py_ext
        ],
        tooltip="Choose a Python file on the client containing layer definitions."
    )
    python_layer_server_file = utils.forms.StringField(
        u'Server-side file',
        validators=[
            validate_file_exists,
            validate_py_ext
        ],
        tooltip="Choose a Python file on the server containing layer definitions."
    )

    train_epochs = utils.forms.IntegerField(
        'Training epochs',
        validators=[
            validators.NumberRange(min=1)
        ],
        default=30,
        tooltip="How many passes through the training data?"
    )

    snapshot_interval = utils.forms.FloatField(
        'Snapshot interval (in epochs)',
        default=1,
        validators=[
            validators.NumberRange(min=0),
        ],
        tooltip="How many epochs of training between taking a snapshot?"
    )

    val_interval = utils.forms.FloatField(
        'Validation interval (in epochs)',
        default=1,
        validators=[
            validators.NumberRange(min=0)
        ],
        tooltip="How many epochs of training between running through one pass of the validation data?"
    )

    random_seed = utils.forms.IntegerField(
        'Random seed',
        validators=[
            validators.NumberRange(min=0),
            validators.Optional(),
        ],
        tooltip=('If you provide a random seed, then back-to-back runs with '
                 'the same model and dataset should give identical results.')
    )

    batch_size = utils.forms.MultiIntegerField(
        'Batch size',
        validators=[
            utils.forms.MultiNumberRange(min=1),
            utils.forms.MultiOptional(),
        ],
        tooltip="How many images to process at once. If blank, values are used from the network definition."
    )

    batch_accumulation = utils.forms.IntegerField(
        'Batch Accumulation',
        validators=[
            validators.NumberRange(min=1),
            validators.Optional(),
        ],
        tooltip=("Accumulate gradients over multiple batches (useful when you "
                 "need a bigger batch size for training but it doesn't fit in memory).")
    )

    # Solver types

    solver_type = utils.forms.SelectField(
        'Solver type',
        choices=[
            ('SGD', 'Stochastic gradient descent (SGD)'),
            ('NESTEROV', "Nesterov's accelerated gradient (NAG)"),
            ('ADAGRAD', 'Adaptive gradient (AdaGrad)'),
            ('RMSPROP', 'RMSprop'),
            ('ADADELTA', 'AdaDelta'),
            ('ADAM', 'Adam'),
        ],
        default='SGD',
        tooltip="What type of solver will be used?",
    )

    def validate_solver_type(form, field):
        fw = frameworks.get_framework_by_id(form.framework)
        if fw is not None:
            if not fw.supports_solver_type(field.data):
                raise validators.ValidationError(
                    'Solver type not supported by this framework')

    # Additional settings specific to selected solver

    rms_decay = utils.forms.FloatField(
        'RMS decay value',
        default=0.99,
        validators=[
            validators.NumberRange(min=0),
        ],
        tooltip=("If the gradient updates results in oscillations the gradient is reduced "
                 "by times 1-rms_decay. Otherwise it will be increased by rms_decay.")
    )

    # Learning rate

    learning_rate = utils.forms.MultiFloatField(
        'Base Learning Rate',
        default=0.01,
        validators=[
            utils.forms.MultiNumberRange(min=0),
        ],
        tooltip=("Affects how quickly the network learns. If you are getting "
                 "NaN for your loss, you probably need to lower this value.")
    )

    lr_policy = wtforms.SelectField(
        'Policy',
        choices=[
            ('fixed', 'Fixed'),
            ('step', 'Step Down'),
            ('multistep', 'Step Down (arbitrary steps)'),
            ('exp', 'Exponential Decay'),
            ('inv', 'Inverse Decay'),
            ('poly', 'Polynomial Decay'),
            ('sigmoid', 'Sigmoid Decay'),
        ],
        default='step'
    )

    lr_step_size = wtforms.FloatField('Step Size', default=33)
    lr_step_gamma = wtforms.FloatField('Gamma', default=0.1)
    lr_multistep_values = wtforms.StringField('Step Values', default="50,85")

    def validate_lr_multistep_values(form, field):
        if form.lr_policy.data == 'multistep':
            for value in field.data.split(','):
                try:
                    float(value)
                except ValueError:
                    raise validators.ValidationError('invalid value')

    lr_multistep_gamma = wtforms.FloatField('Gamma', default=0.5)
    lr_exp_gamma = wtforms.FloatField('Gamma', default=0.95)
    lr_inv_gamma = wtforms.FloatField('Gamma', default=0.1)
    lr_inv_power = wtforms.FloatField('Power', default=0.5)
    lr_poly_power = wtforms.FloatField('Power', default=3)
    lr_sigmoid_step = wtforms.FloatField('Step', default=50)
    lr_sigmoid_gamma = wtforms.FloatField('Gamma', default=0.1)

    # Network

    # Use a SelectField instead of a HiddenField so that the default value
    # is used when nothing is provided (through the REST API)
    method = wtforms.SelectField(
        u'Network type',
        choices=[
            ('standard', 'Standard network'),
            ('previous', 'Previous network'),
            ('pretrained', 'Pretrained network'),
            ('custom', 'Custom network'),
        ],
        default='standard',
    )

    # framework - hidden field, set by Javascript to the selected framework ID
    framework = wtforms.HiddenField(
        'framework',
        validators=[
            validators.AnyOf(
                [fw.get_id() for fw in frameworks.get_frameworks()],
                message='The framework you choose is not currently supported.'
            )
        ],
        default=frameworks.get_frameworks()[0].get_id()
    )

    # The options for this get set in the view (since they are dependent on the data type)
    standard_networks = wtforms.RadioField(
        'Standard Networks',
        validators=[
            validate_required_iff(method='standard'),
        ],
    )

    previous_networks = wtforms.RadioField(
        'Previous Networks',
        choices=[],
        validators=[
            validate_required_iff(method='previous'),
            selection_exists_in_choices,
        ],
    )

    pretrained_networks = wtforms.RadioField(
        'Pretrained Networks',
        choices=[],
        validators=[
            validate_required_iff(method='pretrained'),
            selection_exists_in_choices,
        ],
    )

    custom_network = utils.forms.TextAreaField(
        'Custom Network',
        validators=[
            validate_required_iff(method='custom'),
            validate_NetParameter,
        ],
    )

    custom_network_snapshot = utils.forms.TextField(
        'Pretrained model(s)',
        tooltip=("Paths to pretrained model files, separated by '%s'. "
                 "Only edit this field if you understand how fine-tuning "
                 "works in caffe or torch." % os.path.pathsep)
    )

    def validate_custom_network_snapshot(form, field):
        if form.method.data == 'custom':
            for filename in field.data.strip().split(os.path.pathsep):
                if filename and not os.path.exists(filename):
                    raise validators.ValidationError('File "%s" does not exist' % filename)

    # Select one of several GPUs
    select_gpu = wtforms.RadioField(
        'Select which GPU you would like to use',
        choices=[('next', 'Next available')] + [(
            index,
            '#%s - %s (%s memory)' % (
                index,
                get_device(index).name,
                sizeof_fmt(
                    get_nvml_info(index)['memory']['total']
                    if get_nvml_info(index) and 'memory' in get_nvml_info(index)
                    else get_device(index).totalGlobalMem)
            ),
        ) for index in config_value('gpu_list').split(',') if index],
        default='next',
    )

    # Select N of several GPUs
    select_gpus = utils.forms.SelectMultipleField(
        'Select which GPU[s] you would like to use',
        choices=[(
            index,
            '#%s - %s (%s memory)' % (
                index,
                get_device(index).name,
                sizeof_fmt(
                    get_nvml_info(index)['memory']['total']
                    if get_nvml_info(index) and 'memory' in get_nvml_info(index)
                    else get_device(index).totalGlobalMem)
            ),
        ) for index in config_value('gpu_list').split(',') if index],
        tooltip="The job won't start until all of the chosen GPUs are available."
    )

    # XXX For testing
    # The Flask test framework can't handle SelectMultipleFields correctly
    select_gpus_list = wtforms.StringField('Select which GPU[s] you would like to use (comma separated)')

    def validate_select_gpus(form, field):
        if form.select_gpus_list.data:
            field.data = form.select_gpus_list.data.split(',')

    # Use next available N GPUs
    select_gpu_count = wtforms.IntegerField('Use this many GPUs (next available)',
                                            validators=[
                                                validators.NumberRange(min=1, max=len(
                                                    config_value('gpu_list').split(',')))
                                            ],
                                            default=1,
                                            )

    def validate_select_gpu_count(form, field):
        if field.data is None:
            if form.select_gpus.data:
                # Make this field optional
                field.errors[:] = []
                raise validators.StopValidation()

    model_name = utils.forms.StringField('Model Name',
                                         validators=[
                                             validators.DataRequired()
                                         ],
                                         tooltip="An identifier, later used to refer to this model in the Application."
                                         )

    group_name = utils.forms.StringField('Group Name',
                                         tooltip="An optional group name for organization on the main page."
                                         )

    # allows shuffling data during training (for frameworks that support this, as indicated by
    # their Framework.can_shuffle_data() method)
    shuffle = utils.forms.BooleanField('Shuffle Train Data',
                                       default=True,
                                       tooltip='For every epoch, shuffle the data before training.'
                                       )
