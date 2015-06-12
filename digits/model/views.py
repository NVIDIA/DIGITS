# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import io
import re
import json
import math
import tarfile
import zipfile

import flask
import werkzeug.exceptions
from google.protobuf import text_format
try:
    import caffe_pb2
except ImportError:
    # See issue #32
    from caffe.proto import caffe_pb2
import caffe.draw

import digits
from digits.webapp import app, scheduler, autodoc
from digits.utils.routing import request_wants_json
import images.views
import images as model_images

NAMESPACE = '/models/'

@app.route(NAMESPACE + '<job_id>.json', methods=['GET'])
@app.route(NAMESPACE + '<job_id>', methods=['GET'])
@autodoc(['models', 'api'])
def models_show(job_id):
    """
    Show a ModelJob

    Returns JSON when requested:
        {id, name, directory, status, snapshots: [epoch,epoch,...]}
    """
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    if request_wants_json():
        return flask.jsonify(job.json_dict(True))
    else:
        if isinstance(job, model_images.ImageClassificationModelJob):
            return model_images.classification.views.show(job)
        else:
            raise werkzeug.exceptions.BadRequest(
                    'Invalid job type')

@app.route(NAMESPACE + 'customize', methods=['POST'])
@autodoc('models')
def models_customize():
    """
    Returns a customized file for the ModelJob based on completed form fields
    """
    network = flask.request.args['network']
    if not network:
        raise werkzeug.exceptions.BadRequest('network not provided')

    networks_dir = os.path.join(os.path.dirname(digits.__file__), 'standard-networks')
    for filename in os.listdir(networks_dir):
        path = os.path.join(networks_dir, filename)
        if os.path.isfile(path):
            match = re.match(r'%s.prototxt' % network, filename)
            if match:
                with open(path) as infile:
                    return json.dumps({'network': infile.read()})
    job = scheduler.get_job(network)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    snapshot = None
    try:
        epoch = int(flask.request.form['snapshot_epoch'])
        for filename, e in job.train_task().snapshots:
            if e == epoch:
                snapshot = job.path(filename)
                break
    except:
        pass

    return json.dumps({
        'network': text_format.MessageToString(job.train_task().network),
        'snapshot': snapshot
        })

@app.route(NAMESPACE + 'visualize-network', methods=['POST'])
@autodoc('models')
def models_visualize_network():
    """
    Returns a visualization of the custom network as a string of PNG data
    """
    net = caffe_pb2.NetParameter()
    text_format.Merge(flask.request.form['custom_network'], net)
    # Throws an error if name is None
    if not net.name:
        net.name = 'Network'
    return '<image src="data:image/png;base64,' + caffe.draw.draw_net(net, 'UD').encode('base64') + '" style="max-width:100%" />'

@app.route(NAMESPACE + 'visualize-lr', methods=['POST'])
@autodoc('models')
def models_visualize_lr():
    """
    Returns a JSON object of data used to create the learning rate graph
    """
    policy = flask.request.form['lr_policy']
    lr = float(flask.request.form['learning_rate'])
    if policy == 'fixed':
        pass
    elif policy == 'step':
        step = int(flask.request.form['lr_step_size'])
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

    data = ['Learning Rate']
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

    return json.dumps({'data': {'columns': [data]}})

@app.route(NAMESPACE + '<job_id>/download',
        methods=['GET', 'POST'],
        defaults={'extension': 'tar.gz'})
@app.route(NAMESPACE + '<job_id>/download.<extension>',
        methods=['GET', 'POST'])
@autodoc('models')
def models_download(job_id, extension):
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


