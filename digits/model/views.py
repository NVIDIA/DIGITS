# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import json
import math

from google.protobuf import text_format
from flask import render_template, request, redirect, url_for, flash, make_response, abort
from caffe.proto import caffe_pb2
import caffe.draw

import digits
from digits import utils
from digits.webapp import app, scheduler
from forms import ModelForm
import images.views
import images as model_images

NAMESPACE = '/models/'

### CRUD Routes

@app.route(NAMESPACE + 'new', methods=['GET'])
def models_new():
    form = ModelForm()
    return render_template('models/new.html', form=form)

@app.route(NAMESPACE, methods=['POST'])
def models_create():
    form = ModelForm()
    if form.validate_on_submit():
        return 'Yay!'
    else:
        return render_template('models/new.html', form=form)

@app.route(NAMESPACE + '<job_id>', methods=['GET'])
def models_show(job_id):
    job = scheduler.get_job(job_id)

    if job is None:
        abort(404)

    if isinstance(job, model_images.ImageClassificationModelJob):
        return model_images.classification.views.show(job)
    else:
        abort(404)

### Other routes

@app.route(NAMESPACE + 'customize', methods=['POST'])
def models_customize():
    """Returns a customized file for the Model based on completed form fields"""
    network = request.args.get('network')
    if not network:
        return 'args.network not found!', 400

    networks_dir = os.path.join(os.path.dirname(digits.__file__), 'standard-networks')
    for filename in os.listdir(networks_dir):
        path = os.path.join(networks_dir, filename)
        if os.path.isfile(path):
            match = re.match(r'%s.prototxt' % network, filename)
            if match:
                with open(path) as infile:
                    return json.dumps({'network': infile.read()})
    job = scheduler.get_job(network)
    try:
        epoch = int(request.form['snapshot_epoch'])
        for filename, e in job.train_task().snapshots:
            if e == epoch:
                snapshot = job.path(filename)
                break
    except Exception as e:
        snapshot = None
    if job:
        return json.dumps({
            'network': text_format.MessageToString(job.train_task().network),
            'snapshot': snapshot
            })

    return 'ERROR: Network not found!', 400

@app.route(NAMESPACE + 'visualize-network', methods=['POST'])
def models_visualize_network():
    """Returns a string of png data"""
    net = caffe_pb2.NetParameter()
    text_format.Merge(request.form['custom_network'], net)
    # Throws an error if name is None
    if not net.name:
        net.name = 'Network'
    return '<image src="data:image/png;base64,' + caffe.draw.draw_net(net, 'UD').encode('base64') + '" style="max-width:100%" />'

@app.route(NAMESPACE + 'visualize-lr', methods=['POST'])
def models_visualize_lr():
    policy = request.form['lr_policy']
    lr = float(request.form['learning_rate'])
    data = [('I', 'Learning Rate')]
    if policy == 'fixed':
        pass
    elif policy == 'step':
        step = int(request.form['lr_step_size'])
        gamma = float(request.form['lr_step_gamma'])
    elif policy == 'multistep':
        steps = [float(s) for s in request.form['lr_multistep_values'].split(',')]
        current_step = 0
        gamma = float(request.form['lr_multistep_gamma'])
    elif policy == 'exp':
        gamma = float(request.form['lr_exp_gamma'])
    elif policy == 'inv':
        gamma = float(request.form['lr_inv_gamma'])
        power = float(request.form['lr_inv_power'])
    elif policy == 'poly':
        power = float(request.form['lr_poly_power'])
    elif policy == 'sigmoid':
        step = float(request.form['lr_sigmoid_step'])
        gamma = float(request.form['lr_sigmoid_gamma'])
    else:
        return 'Invalid policy', 404

    for i in xrange(100):
        if policy == 'fixed':
            data.append((i, lr))
        elif policy == 'step':
            data.append((i, lr * math.pow(gamma, math.floor(float(i)/step))))
        elif policy == 'multistep':
            if current_step < len(steps) and i >= steps[current_step]:
                current_step += 1
            data.append((i, lr * math.pow(gamma, current_step)))
        elif policy == 'exp':
            data.append((i, lr * math.pow(gamma, i)))
        elif policy == 'inv':
            data.append((i, lr * math.pow(1.0 + gamma * i, -power)))
        elif policy == 'poly':
            data.append((i, lr * math.pow(1.0 - float(i)/100, power)))
        elif policy == 'sigmoid':
            data.append((i, lr / (1.0 + math.exp(gamma * (i - step)))))

    return json.dumps(data)

@app.route(NAMESPACE + '<job_id>/download_snapshot', methods=['POST'])
def models_download_snapshot(job_id):
    job = scheduler.get_job(job_id)

    if not job:
        abort(404)

    epoch = int(request.form['snapshot_epoch'])
    filename = None

    for f, e in job.train_task().snapshots:
        if e == epoch:
            filename = f
            break
    if not filename:
        abort(400)

    with open(job.path(filename), 'r') as infile:
        response = make_response(infile.read())
        response.headers["Content-Disposition"] = "attachment; filename=%s_epoch_%s.caffemodel" % (job.id(), epoch)
        return response

