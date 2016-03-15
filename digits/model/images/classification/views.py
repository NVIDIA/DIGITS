# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

import flask
import numpy as np
import werkzeug.exceptions

from .forms import ImageClassificationModelForm
from .job import ImageClassificationModelJob
import digits
from digits import frameworks
from digits import utils
from digits.config import config_value
from digits.dataset import ImageClassificationDatasetJob
from digits.status import Status
from digits.utils import filesystem as fs
from digits.utils.forms import fill_form_if_cloned, save_form_to_job
from digits.utils.routing import request_wants_json, job_from_request
from digits.webapp import app, scheduler

blueprint = flask.Blueprint(__name__, __name__)

@blueprint.route('/new', methods=['GET'])
@utils.auth.requires_login
def new():
    """
    Return a form for a new ImageClassificationModelJob
    """
    form = ImageClassificationModelForm()
    form.dataset.choices = get_datasets()
    form.standard_networks.choices = get_standard_networks()
    form.standard_networks.default = get_default_standard_network()
    form.previous_networks.choices = get_previous_networks()

    prev_network_snapshots = get_previous_network_snapshots()

    ## Is there a request to clone a job with ?clone=<job_id>
    fill_form_if_cloned(form)

    return flask.render_template('models/images/classification/new.html',
            form = form,
            frameworks = frameworks.get_frameworks(),
            previous_network_snapshots = prev_network_snapshots,
            previous_networks_fullinfo = get_previous_networks_fulldetails(),
            multi_gpu = config_value('caffe_root')['multi_gpu'],
            )

@blueprint.route('.json', methods=['POST'])
@blueprint.route('', methods=['POST'], strict_slashes=False)
@utils.auth.requires_login(redirect=False)
def create():
    """
    Create a new ImageClassificationModelJob

    Returns JSON when requested: {job_id,name,status} or {errors:[]}
    """
    form = ImageClassificationModelForm()
    form.dataset.choices = get_datasets()
    form.standard_networks.choices = get_standard_networks()
    form.standard_networks.default = get_default_standard_network()
    form.previous_networks.choices = get_previous_networks()

    prev_network_snapshots = get_previous_network_snapshots()

    ## Is there a request to clone a job with ?clone=<job_id>
    fill_form_if_cloned(form)

    if not form.validate_on_submit():
        if request_wants_json():
            return flask.jsonify({'errors': form.errors}), 400
        else:
            return flask.render_template('models/images/classification/new.html',
                    form = form,
                    frameworks = frameworks.get_frameworks(),
                    previous_network_snapshots = prev_network_snapshots,
                    previous_networks_fullinfo = get_previous_networks_fulldetails(),
                    multi_gpu = config_value('caffe_root')['multi_gpu'],
                    ), 400

    datasetJob = scheduler.get_job(form.dataset.data)
    if not datasetJob:
        raise werkzeug.exceptions.BadRequest(
                'Unknown dataset job_id "%s"' % form.dataset.data)

    job = None
    try:
        job = ImageClassificationModelJob(
                username    = utils.auth.get_username(),
                name        = form.model_name.data,
                dataset_id  = datasetJob.id(),
                )
        # get handle to framework object
        fw = frameworks.get_framework_by_id(form.framework.data)

        pretrained_model = None
        if form.method.data == 'standard':
            found = False

            # can we find it in standard networks?
            network_desc = fw.get_standard_network_desc(form.standard_networks.data)
            if network_desc:
                found = True
                network = fw.get_network_from_desc(network_desc)

            if not found:
                raise werkzeug.exceptions.BadRequest(
                        'Unknown standard model "%s"' % form.standard_networks.data)
        elif form.method.data == 'previous':
            old_job = scheduler.get_job(form.previous_networks.data)
            if not old_job:
                raise werkzeug.exceptions.BadRequest(
                        'Job not found: %s' % form.previous_networks.data)

            use_same_dataset = (old_job.dataset_id == job.dataset_id)
            network = fw.get_network_from_previous(old_job.train_task().network, use_same_dataset)

            for choice in form.previous_networks.choices:
                if choice[0] == form.previous_networks.data:
                    epoch = float(flask.request.form['%s-snapshot' % form.previous_networks.data])
                    if epoch == 0:
                        pass
                    elif epoch == -1:
                        pretrained_model = old_job.train_task().pretrained_model
                    else:
                        for filename, e in old_job.train_task().snapshots:
                            if e == epoch:
                                pretrained_model = filename
                                break

                        if pretrained_model is None:
                            raise werkzeug.exceptions.BadRequest(
                                    "For the job %s, selected pretrained_model for epoch %d is invalid!"
                                    % (form.previous_networks.data, epoch))
                        if not (os.path.exists(pretrained_model)):
                            raise werkzeug.exceptions.BadRequest(
                                    "Pretrained_model for the selected epoch doesn't exists. May be deleted by another user/process. Please restart the server to load the correct pretrained_model details")
                    break

        elif form.method.data == 'custom':
            network = fw.get_network_from_desc(form.custom_network.data)
            pretrained_model = form.custom_network_snapshot.data.strip()
        else:
            raise werkzeug.exceptions.BadRequest(
                    'Unrecognized method: "%s"' % form.method.data)

        policy = {'policy': form.lr_policy.data}
        if form.lr_policy.data == 'fixed':
            pass
        elif form.lr_policy.data == 'step':
            policy['stepsize'] = form.lr_step_size.data
            policy['gamma'] = form.lr_step_gamma.data
        elif form.lr_policy.data == 'multistep':
            policy['stepvalue'] = form.lr_multistep_values.data
            policy['gamma'] = form.lr_multistep_gamma.data
        elif form.lr_policy.data == 'exp':
            policy['gamma'] = form.lr_exp_gamma.data
        elif form.lr_policy.data == 'inv':
            policy['gamma'] = form.lr_inv_gamma.data
            policy['power'] = form.lr_inv_power.data
        elif form.lr_policy.data == 'poly':
            policy['power'] = form.lr_poly_power.data
        elif form.lr_policy.data == 'sigmoid':
            policy['stepsize'] = form.lr_sigmoid_step.data
            policy['gamma'] = form.lr_sigmoid_gamma.data
        else:
            raise werkzeug.exceptions.BadRequest(
                    'Invalid learning rate policy')

        if config_value('caffe_root')['multi_gpu']:
            if form.select_gpus.data:
                selected_gpus = [str(gpu) for gpu in form.select_gpus.data]
                gpu_count = None
            elif form.select_gpu_count.data:
                gpu_count = form.select_gpu_count.data
                selected_gpus = None
            else:
                gpu_count = 1
                selected_gpus = None
        else:
            if form.select_gpu.data == 'next':
                gpu_count = 1
                selected_gpus = None
            else:
                selected_gpus = [str(form.select_gpu.data)]
                gpu_count = None

        # Python Layer File may be on the server or copied from the client.
        fs.copy_python_layer_file(
            bool(form.python_layer_from_client.data),
            job.dir(),
            (flask.request.files[form.python_layer_client_file.name]
             if form.python_layer_client_file.name in flask.request.files
             else ''), form.python_layer_server_file.data)

        job.tasks.append(fw.create_train_task(
                    job_dir         = job.dir(),
                    dataset         = datasetJob,
                    train_epochs    = form.train_epochs.data,
                    snapshot_interval   = form.snapshot_interval.data,
                    learning_rate   = form.learning_rate.data,
                    lr_policy       = policy,
                    gpu_count       = gpu_count,
                    selected_gpus   = selected_gpus,
                    batch_size      = form.batch_size.data,
                    val_interval    = form.val_interval.data,
                    pretrained_model= pretrained_model,
                    crop_size       = form.crop_size.data,
                    use_mean        = form.use_mean.data,
                    network         = network,
                    random_seed     = form.random_seed.data,
                    solver_type     = form.solver_type.data,
                    shuffle         = form.shuffle.data,
                    )
                )

        ## Save form data with the job so we can easily clone it later.
        save_form_to_job(job, form)

        scheduler.add_job(job)
        if request_wants_json():
            return flask.jsonify(job.json_dict())
        else:
            return flask.redirect(flask.url_for('digits.model.views.show', job_id=job.id()))

    except:
        if job:
            scheduler.delete_job(job)
        raise

def show(job):
    """
    Called from digits.model.views.models_show()
    """
    return flask.render_template('models/images/classification/show.html', job=job, framework_ids = [fw.get_id() for fw in frameworks.get_frameworks()])

@blueprint.route('/large_graph', methods=['GET'])
def large_graph():
    """
    Show the loss/accuracy graph, but bigger
    """
    job = job_from_request()

    return flask.render_template('models/images/classification/large_graph.html', job=job)

def get_datasets():
    return [(j.id(), j.name()) for j in sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, ImageClassificationDatasetJob) and (j.status.is_running() or j.status == Status.DONE)],
        cmp=lambda x,y: cmp(y.id(), x.id())
        )
        ]

def get_standard_networks():
    return [
            ('lenet', 'LeNet'),
            ('alexnet', 'AlexNet'),
            #('vgg-16', 'VGG (16-layer)'), #XXX model won't learn
            ('googlenet', 'GoogLeNet'),
            ]

def get_default_standard_network():
    return 'alexnet'

def get_previous_networks():
    return [(j.id(), j.name()) for j in sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, ImageClassificationModelJob)],
        cmp=lambda x,y: cmp(y.id(), x.id())
        )
        ]

def get_previous_networks_fulldetails():
    return [(j) for j in sorted(
        [j for j in scheduler.jobs.values() if isinstance(j, ImageClassificationModelJob)],
        cmp=lambda x,y: cmp(y.id(), x.id())
        )
        ]

def get_previous_network_snapshots():
    prev_network_snapshots = []
    for job_id, _ in get_previous_networks():
        job = scheduler.get_job(job_id)
        e = [(0, 'None')] + [(epoch, 'Epoch #%s' % epoch)
                for _, epoch in reversed(job.train_task().snapshots)]
        if job.train_task().pretrained_model:
            e.insert(0, (-1, 'Previous pretrained model'))
        prev_network_snapshots.append(e)
    return prev_network_snapshots

