# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import tempfile

import flask
import werkzeug.exceptions

from digits import frameworks
from digits.config import config_value
from digits import utils
from digits.utils.routing import request_wants_json, job_from_request
from digits.utils.forms import fill_form_if_cloned, save_form_to_job
from digits.webapp import app, scheduler, autodoc
from digits.dataset import GenericImageDatasetJob
from digits import frameworks
from digits.model import tasks
from forms import GenericImageModelForm
from job import GenericImageModelJob
from digits.status import Status
import platform
from digits.utils import filesystem as fs

NAMESPACE   = '/models/images/generic'

@app.route(NAMESPACE + '/new', methods=['GET'])
@autodoc('models')
def generic_image_model_new():
    """
    Return a form for a new GenericImageModelJob
    """
    form = GenericImageModelForm()
    form.dataset.choices = get_datasets()
    form.standard_networks.choices = []
    form.previous_networks.choices = get_previous_networks()

    prev_network_snapshots = get_previous_network_snapshots()

    ## Is there a request to clone a job with ?clone=<job_id>
    fill_form_if_cloned(form)

    return flask.render_template('models/images/generic/new.html',
            form = form,
            frameworks = frameworks.get_frameworks(),
            previous_network_snapshots = prev_network_snapshots,
            previous_networks_fullinfo = get_previous_networks_fulldetails(),
            multi_gpu = config_value('caffe_root')['multi_gpu'],
            )

@app.route(NAMESPACE + '.json', methods=['POST'])
@app.route(NAMESPACE, methods=['POST'])
@autodoc(['models', 'api'])
def generic_image_model_create():
    """
    Create a new GenericImageModelJob

    Returns JSON when requested: {job_id,name,status} or {errors:[]}
    """
    form = GenericImageModelForm()
    form.dataset.choices = get_datasets()
    form.standard_networks.choices = []
    form.previous_networks.choices = get_previous_networks()

    prev_network_snapshots = get_previous_network_snapshots()

    ## Is there a request to clone a job with ?clone=<job_id>
    fill_form_if_cloned(form)

    if not form.validate_on_submit():
        if request_wants_json():
            return flask.jsonify({'errors': form.errors}), 400
        else:
            return flask.render_template('models/images/generic/new.html',
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
        job = GenericImageModelJob(
                name        = form.model_name.data,
                dataset_id  = datasetJob.id(),
                )

        # get framework (hard-coded to caffe for now)
        fw = frameworks.get_framework_by_id(form.framework.data)

        pretrained_model = None
        #if form.method.data == 'standard':
        if form.method.data == 'previous':
            old_job = scheduler.get_job(form.previous_networks.data)
            if not old_job:
                raise werkzeug.exceptions.BadRequest(
                        'Job not found: %s' % form.previous_networks.data)

            network = fw.get_network_from_previous(old_job.train_task().network)

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
            if form.select_gpu_count.data:
                gpu_count = form.select_gpu_count.data
                selected_gpus = None
            else:
                selected_gpus = [str(gpu) for gpu in form.select_gpus.data]
                gpu_count = None
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
            return flask.redirect(flask.url_for('models_show', job_id=job.id()))

    except:
        if job:
            scheduler.delete_job(job)
        raise

def show(job):
    """
    Called from digits.model.views.models_show()
    """
    return flask.render_template('models/images/generic/show.html', job=job)

@app.route(NAMESPACE + '/large_graph', methods=['GET'])
@autodoc('models')
def generic_image_model_large_graph():
    """
    Show the loss/accuracy graph, but bigger
    """
    job = job_from_request()

    return flask.render_template('models/images/generic/large_graph.html', job=job)

@app.route(NAMESPACE + '/infer_one.json', methods=['POST'])
@app.route(NAMESPACE + '/infer_one', methods=['POST', 'GET'])
@autodoc(['models', 'api'])
def generic_image_model_infer_one():
    """
    Infer one image
    """
    job = job_from_request()

    image = None
    if 'image_url' in flask.request.form and flask.request.form['image_url']:
        image = utils.image.load_image(flask.request.form['image_url'])
    elif 'image_file' in flask.request.files and flask.request.files['image_file']:
        outfile = tempfile.mkstemp(suffix='.bin')
        flask.request.files['image_file'].save(outfile[1])
        image = utils.image.load_image(outfile[1])
        os.close(outfile[0])
        os.remove(outfile[1])
    else:
        raise werkzeug.exceptions.BadRequest('must provide image_url or image_file')

    # resize image
    db_task = job.train_task().dataset.analyze_db_tasks()[0]
    height = db_task.image_height
    width = db_task.image_width
    image = utils.image.resize_image(image, height, width,
            channels = db_task.image_channels,
            resize_mode = 'squash',
            )

    epoch = None
    if 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])

    layers = 'none'
    if 'show_visualizations' in flask.request.form and flask.request.form['show_visualizations']:
        layers = 'all'

    outputs, visualizations = job.train_task().infer_one(image, snapshot_epoch=epoch, layers=layers)

    if request_wants_json():
        return flask.jsonify({'outputs': dict((name, blob.tolist()) for name,blob in outputs.iteritems())})
    else:
        return flask.render_template('models/images/generic/infer_one.html',
                job             = job,
                image_src       = utils.image.embed_image_html(image),
                network_outputs = outputs,
                visualizations  = visualizations,
                total_parameters= sum(v['param_count'] for v in visualizations if v['vis_type'] == 'Weights'),
                )

@app.route(NAMESPACE + '/infer_many.json', methods=['POST'])
@app.route(NAMESPACE + '/infer_many', methods=['POST', 'GET'])
@autodoc(['models', 'api'])
def generic_image_model_infer_many():
    """
    Infer many images
    """
    job = job_from_request()

    image_list = flask.request.files.get('image_list')
    if not image_list:
        raise werkzeug.exceptions.BadRequest('image_list is a required field')

    epoch = None
    if 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])

    paths = []
    images = []

    db_task = job.train_task().dataset.analyze_db_tasks()[0]
    height = db_task.image_height
    width = db_task.image_width
    channels = db_task.image_channels

    for line in image_list.readlines():
        line = line.strip()
        if not line:
            continue

        path = None
        # might contain a numerical label at the end
        match = re.match(r'(.*\S)\s+\d+$', line)
        if match:
            path = match.group(1)
        else:
            path = line

        try:
            image = utils.image.load_image(path)
            image = utils.image.resize_image(image, height, width,
                    channels = channels,
                    resize_mode = 'squash',
                    )
            paths.append(path)
            images.append(image)
        except utils.errors.LoadImageError as e:
            print e

    if not len(images):
        raise werkzeug.exceptions.BadRequest(
                'Unable to load any images from the file')

    outputs = job.train_task().infer_many(images, snapshot_epoch=epoch)
    if outputs is None:
        raise RuntimeError('An error occured while processing the images')

    if request_wants_json():
        result = {}
        for i, path in enumerate(paths):
            result[path] = dict((name, blob[i].tolist()) for name,blob in outputs.iteritems())
        return flask.jsonify({'outputs': result})
    else:
        return flask.render_template('models/images/generic/infer_many.html',
                job             = job,
                paths           = paths,
                network_outputs = outputs,
                )

def get_datasets():
    return [(j.id(), j.name()) for j in sorted(
        [j for j in scheduler.jobs if isinstance(j, GenericImageDatasetJob) and (j.status.is_running() or j.status == Status.DONE)],
        cmp=lambda x,y: cmp(y.id(), x.id())
        )
        ]

def get_previous_networks():
    return [(j.id(), j.name()) for j in sorted(
        [j for j in scheduler.jobs if isinstance(j, GenericImageModelJob)],
        cmp=lambda x,y: cmp(y.id(), x.id())
        )
        ]

def get_previous_networks_fulldetails():
    return [(j) for j in sorted(
        [j for j in scheduler.jobs if isinstance(j, GenericImageModelJob)],
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

