# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from datetime import timedelta
import io
import json
import math
import tarfile
import zipfile

import flask
import werkzeug.exceptions

from . import images as model_images
from . import ModelJob
from digits import frameworks, extensions
from digits.utils import time_filters
from digits.utils.routing import request_wants_json
from digits.webapp import scheduler

blueprint = flask.Blueprint(__name__, __name__)

@blueprint.route('', methods=['GET'])
def index():
    column_attrs = list(get_column_attrs())
    raw_jobs = [j for j in scheduler.jobs.values() if isinstance(j, ModelJob)]

    column_types = [
        ColumnType('latest', False, lambda outs: outs[-1]),
        ColumnType('max', True, lambda outs: max(outs)),
        ColumnType('min', True, lambda outs: min(outs))
    ]

    jobs = []
    for rjob in raw_jobs:
        train_outs = rjob.train_task().train_outputs
        val_outs = rjob.train_task().val_outputs
        history = rjob.status_history

        # update column attribute set
        keys = set(train_outs.keys() + val_outs.keys())

        # build job dict
        job_info = JobBasicInfo(
            rjob.name(),
            rjob.id(),
            rjob.status,
            time_filters.print_time_diff_nosuffixes(history[-1][1] - history[0][1]),
            rjob.train_task().framework_id
        )

        # build a dictionary of each attribute of a job. If an attribute is
        # present, add all different column types.
        job_attrs = {}
        for cattr in column_attrs:
            if cattr in train_outs:
                out_list = train_outs[cattr].data
            elif cattr in val_outs:
                out_list = val_outs[cattr].data
            else:
                continue

            job_attrs[cattr] = {ctype.name: ctype.find_from_list(out_list)
                for ctype in column_types}

        job = (job_info, job_attrs)
        jobs.append(job)

    attrs_and_labels = []
    for cattr in column_attrs:
        for ctype in column_types:
            attrs_and_labels.append((cattr, ctype, ctype.label(cattr)))

    return flask.render_template('models/index.html',
        jobs=jobs,
        attrs_and_labels=attrs_and_labels)

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

    return json.dumps({
            'network': job.train_task().get_network_desc(),
            'snapshot': snapshot
            })


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

    fw = frameworks.get_framework_by_id(framework)
    ret = fw.get_network_visualization(flask.request.form['custom_network'])

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
                data.append(lr * math.pow(gamma, math.floor(float(i)/step)))
            elif policy == 'multistep':
                if current_step < len(steps) and i >= steps[current_step]:
                    current_step += 1
                data.append(lr * math.pow(gamma, current_step))
            elif policy == 'exp':
                data.append(lr * math.pow(gamma, i))
            elif policy == 'inv':
                data.append(lr * math.pow(1.0 + gamma * i, -power))
            elif policy == 'poly':
                data.append(lr * math.pow(1.0 - float(i)/100, power))
            elif policy == 'sigmoid':
                data.append(lr / (1.0 + math.exp(gamma * (i - step))))
        datalist.append(data)

    return json.dumps({'data': {'columns': datalist}})

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

    task = job.train_task()

    snapshot_filename = None
    if epoch == -1 and len(task.snapshots):
        epoch = task.snapshots[-1][1]
        snapshot_filename = task.snapshots[-1][0]
    else:
        for f, e in task.snapshots:
            if e == epoch:
                snapshot_filename = f
                break
    if not snapshot_filename:
        raise werkzeug.exceptions.BadRequest('Invalid epoch')

    b = io.BytesIO()
    if extension in ['tar', 'tar.gz', 'tgz', 'tar.bz2']:
        # tar file
        mode = ''
        if extension in ['tar.gz', 'tgz']:
            mode = 'gz'
        elif extension in ['tar.bz2']:
            mode = 'bz2'
        with tarfile.open(fileobj=b, mode='w:%s' % mode) as tf:
            for path, name in job.download_files(epoch):
                tf.add(path, arcname=name)
    elif extension in ['zip']:
        with zipfile.ZipFile(b, 'w') as zf:
            for path, name in job.download_files(epoch):
                zf.write(path, arcname=name)
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
