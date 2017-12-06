# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import io
import json
import math
import os
import tarfile
import zipfile

import flask
from flask import flash
import requests
import werkzeug.exceptions

from . import images as model_images
from . import ModelJob
from digits.pretrained_model.job import PretrainedModelJob
from digits import frameworks, extensions
from digits.utils import auth
from digits.utils.routing import request_wants_json, job_from_request, get_request_arg
from digits.webapp import scheduler

blueprint = flask.Blueprint(__name__, __name__)


@blueprint.route('/<job_id>.json', methods=['GET'])
@blueprint.route('/<job_id>', methods=['GET'])
def show(job_id):
    """
    Show a ModelJob

    Returns JSON when requested:
        {id, name, directory, status, snapshots: [epoch,epoch,...]}
    """
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    related_jobs = scheduler.get_related_jobs(job)

    if request_wants_json():
        return flask.jsonify(job.json_dict(True))
    else:
        if isinstance(job, model_images.ImageClassificationModelJob):
            return model_images.classification.views.show(job, related_jobs=related_jobs)
        elif isinstance(job, model_images.GenericImageModelJob):
            return model_images.generic.views.show(job, related_jobs=related_jobs)
        else:
            raise werkzeug.exceptions.BadRequest(
                'Invalid job type')


@blueprint.route('/customize', methods=['POST'])
def customize():
    """
    Returns a customized file for the ModelJob based on completed form fields
    """
    network = flask.request.args['network']
    framework = flask.request.args.get('framework')
    if not network:
        raise werkzeug.exceptions.BadRequest('network not provided')

    fw = frameworks.get_framework_by_id(framework)

    # can we find it in standard networks?
    network_desc = fw.get_standard_network_desc(network)
    if network_desc:
        return json.dumps({'network': network_desc})

    # not found in standard networks, looking for matching job
    job = scheduler.get_job(network)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    snapshot = None
    epoch = float(flask.request.form.get('snapshot_epoch', 0))
    if epoch == 0:
        pass
    elif epoch == -1:
        snapshot = job.train_task().pretrained_model
    else:

        for filename, e in job.train_task().snapshots:
            if e == epoch:
                snapshot = job.path(filename)
                break

    if isinstance(job, PretrainedModelJob):
        model_def = open(job.get_model_def_path(), 'r')
        network = model_def.read()
        snapshot = job.get_weights_path()
        python_layer = job.get_python_layer_path()
    else:
        network = job.train_task().get_network_desc()
        python_layer = None

    return json.dumps({
        'network': network,
        'snapshot': snapshot,
        'python_layer': python_layer
    })


@blueprint.route('/timeline_trace_data', methods=['POST'])
def timeline_trace_data():
    """
    Shows timeline trace of a model
    """
    job = job_from_request()
    step = get_request_arg('step')
    if step is None:
        raise werkzeug.exceptions.BadRequest('step is a required field')
    return job.train_task().timeline_trace(int(step))


@blueprint.route('/view-config/<extension_id>', methods=['GET'])
def view_config(extension_id):
    """
    Returns a rendering of a view extension configuration template
    """
    extension = extensions.view.get_extension(extension_id)
    if extension is None:
        raise ValueError("Unknown extension '%s'" % extension_id)
    config_form = extension.get_config_form()
    template, context = extension.get_config_template(config_form)
    return flask.render_template_string(template, **context)


@blueprint.route('/visualize-network', methods=['POST'])
def visualize_network():
    """
    Returns a visualization of the custom network as a string of PNG data
    """
    framework = flask.request.args.get('framework')
    if not framework:
        raise werkzeug.exceptions.BadRequest('framework not provided')

    dataset = None
    if 'dataset_id' in flask.request.form:
        dataset = scheduler.get_job(flask.request.form['dataset_id'])

    fw = frameworks.get_framework_by_id(framework)
    ret = fw.get_network_visualization(
        desc=flask.request.form['custom_network'],
        dataset=dataset,
        solver_type=flask.request.form['solver_type'] if 'solver_type' in flask.request.form else None,
        use_mean=flask.request.form['use_mean'] if 'use_mean' in flask.request.form else None,
        crop_size=flask.request.form['crop_size'] if 'crop_size' in flask.request.form else None,
        num_gpus=flask.request.form['num_gpus'] if 'num_gpus' in flask.request.form else None,
    )
    return ret


@blueprint.route('/visualize-lr', methods=['POST'])
def visualize_lr():
    """
    Returns a JSON object of data used to create the learning rate graph
    """
    policy = flask.request.form['lr_policy']
    # There may be multiple lrs if the learning_rate is swept
    lrs = map(float, flask.request.form['learning_rate'].split(','))
    if policy == 'fixed':
        pass
    elif policy == 'step':
        step = float(flask.request.form['lr_step_size'])
        gamma = float(flask.request.form['lr_step_gamma'])
    elif policy == 'multistep':
        steps = [float(s) for s in flask.request.form['lr_multistep_values'].split(',')]
        current_step = 0
        gamma = float(flask.request.form['lr_multistep_gamma'])
    elif policy == 'exp':
        gamma = float(flask.request.form['lr_exp_gamma'])
    elif policy == 'inv':
        gamma = float(flask.request.form['lr_inv_gamma'])
        power = float(flask.request.form['lr_inv_power'])
    elif policy == 'poly':
        power = float(flask.request.form['lr_poly_power'])
    elif policy == 'sigmoid':
        step = float(flask.request.form['lr_sigmoid_step'])
        gamma = float(flask.request.form['lr_sigmoid_gamma'])
    else:
        raise werkzeug.exceptions.BadRequest('Invalid policy')

    datalist = []
    for j, lr in enumerate(lrs):
        data = ['Learning Rate %d' % j]
        for i in xrange(101):
            if policy == 'fixed':
                data.append(lr)
            elif policy == 'step':
                data.append(lr * math.pow(gamma, math.floor(float(i) / step)))
            elif policy == 'multistep':
                if current_step < len(steps) and i >= steps[current_step]:
                    current_step += 1
                data.append(lr * math.pow(gamma, current_step))
            elif policy == 'exp':
                data.append(lr * math.pow(gamma, i))
            elif policy == 'inv':
                data.append(lr * math.pow(1.0 + gamma * i, -power))
            elif policy == 'poly':
                data.append(lr * math.pow(1.0 - float(i) / 100, power))
            elif policy == 'sigmoid':
                data.append(lr / (1.0 + math.exp(gamma * (i - step))))
        datalist.append(data)

    return json.dumps({'data': {'columns': datalist}})


@auth.requires_login
@blueprint.route('/<job_id>/to_pretrained', methods=['GET', 'POST'])
def to_pretrained(job_id):
    job = scheduler.get_job(job_id)

    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    epoch = -1
    # GET ?epoch=n
    if 'epoch' in flask.request.args:
        epoch = float(flask.request.args['epoch'])

    # POST ?snapshot_epoch=n (from form)
    elif 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])

    # Write the stats of the job to json,
    # and store in tempfile (for archive)
    info = job.json_dict(verbose=False, epoch=epoch)

    task = job.train_task()
    snapshot_filename = None
    snapshot_filename = task.get_snapshot(epoch)

    # Set defaults:
    labels_path = None
    resize_mode = None

    if "labels file" in info:
        labels_path = os.path.join(task.dataset.dir(), info["labels file"])
    if "image resize mode" in info:
        resize_mode = info["image resize mode"]

    job = PretrainedModelJob(
        snapshot_filename,
        os.path.join(job.dir(), task.model_file),
        labels_path,
        info["framework"],
        info["image dimensions"][2],
        resize_mode,
        info["image dimensions"][0],
        info["image dimensions"][1],
        username=auth.get_username(),
        name=info["name"]
    )

    scheduler.add_job(job)

    return flask.redirect(flask.url_for('digits.views.home', tab=3)), 302


@blueprint.route('/<job_id>/publish_inference', methods=['POST'])
def publish_inference(job_id):
    """
    Publish model to inference server
    """
    rie_url = os.environ.get('RIE_URL', "http://localhost:5055")

    publish_endpoint = rie_url+'/models'

    # Get data from the modal form
    description = flask.request.form.get('description')
    modality = flask.request.form.getlist('modality')
    output_layer = flask.request.form.get('output_layer')
    input_layer = flask.request.form.get('input_layer')
    input_shape = flask.request.form.get('input_shape')
    output_shape = flask.request.form.get('output_shape')

    job = scheduler.get_job(job_id)

    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    epoch = -1
    # GET ?epoch=n
    if 'epoch' in flask.request.args:
        epoch = float(flask.request.args['epoch'])

    # POST ?snapshot_epoch=n (from form)
    elif 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])

    # Write the stats of the job to json,
    # and store in tempfile (for archive)
    job_dict = job.json_dict(verbose=False, epoch=epoch)
    job_dict.update({"output_layer": output_layer,
                     "description": description,
                     "input_layer": input_layer,
                     "input_shape": input_shape,
                     "output_shape": output_shape,
                     "modality": modality})
    info = json.dumps(job_dict, sort_keys=True, indent=4, separators=(',', ': '))
    info_io = io.BytesIO()
    info_io.write(info)

    b = io.BytesIO()
    mode = ''
    with tarfile.open(fileobj=b, mode='w:%s' % mode) as tar:
        for path, name in job.download_files(epoch, frozen_file=(job_dict['framework'] == 'tensorflow')):
            tar.add(path, arcname=name)
        tar_info = tarfile.TarInfo("info.json")
        tar_info.size = len(info_io.getvalue())
        info_io.seek(0)
        tar.addfile(tar_info, info_io)

    temp_buffer = b.getvalue()
    files = {'model': ('tmp.tgz', temp_buffer)}
    try:
        r = requests.post(publish_endpoint, files=files)
    except Exception as e:
        return flask.make_response(e)
    if r.status_code != requests.codes.ok:
        raise werkzeug.exceptions.BadRequest("Bad Request")
    end_point = json.loads(r.text)["location"]
    flash('Model successfully published to RIE.<p>New endpoint at {}'.format(end_point))
    return flask.redirect(flask.request.referrer), 302


@blueprint.route('/<job_id>/download',
                 methods=['GET', 'POST'],
                 defaults={'extension': 'tar.gz'})
@blueprint.route('/<job_id>/download.<extension>',
                 methods=['GET', 'POST'])
def download(job_id, extension):
    """
    Return a tarball of all files required to run the model
    """

    job = scheduler.get_job(job_id)

    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    epoch = -1
    # GET ?epoch=n
    if 'epoch' in flask.request.args:
        epoch = float(flask.request.args['epoch'])

    # POST ?snapshot_epoch=n (from form)
    elif 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])

    # Write the stats of the job to json,
    # and store in tempfile (for archive)
    info = json.dumps(job.json_dict(verbose=False, epoch=epoch), sort_keys=True, indent=4, separators=(',', ': '))
    info_io = io.BytesIO()
    info_io.write(info)

    b = io.BytesIO()
    if extension in ['tar', 'tar.gz', 'tgz', 'tar.bz2']:
        # tar file
        mode = ''
        if extension in ['tar.gz', 'tgz']:
            mode = 'gz'
        elif extension in ['tar.bz2']:
            mode = 'bz2'
        with tarfile.open(fileobj=b, mode='w:%s' % mode) as tar:
            for path, name in job.download_files(epoch):
                tar.add(path, arcname=name)
            tar_info = tarfile.TarInfo("info.json")
            tar_info.size = len(info_io.getvalue())
            info_io.seek(0)
            tar.addfile(tar_info, info_io)
    elif extension in ['zip']:
        with zipfile.ZipFile(b, 'w') as zf:
            for path, name in job.download_files(epoch):
                zf.write(path, arcname=name)
            zf.writestr("info.json", info_io.getvalue())
    else:
        raise werkzeug.exceptions.BadRequest('Invalid extension')

    response = flask.make_response(b.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=%s_epoch_%s.%s' % (job.id(), epoch, extension)
    return response


class JobBasicInfo(object):

    def __init__(self, name, ID, status, time, framework_id):
        self.name = name
        self.id = ID
        self.status = status
        self.time = time
        self.framework_id = framework_id


class ColumnType(object):

    def __init__(self, name, has_suffix, find_fn):
        self.name = name
        self.has_suffix = has_suffix
        self.find_from_list = find_fn

    def label(self, attr):
        if self.has_suffix:
            return '{} {}'.format(attr, self.name)
        else:
            return attr


def get_column_attrs():
    job_outs = [set(j.train_task().train_outputs.keys() + j.train_task().val_outputs.keys())
                for j in scheduler.jobs.values() if isinstance(j, ModelJob)]

    return reduce(lambda acc, j: acc.union(j), job_outs, set())
