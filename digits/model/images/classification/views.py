# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import sys
import shutil
import tempfile
import random

import numpy as np
from flask import render_template, request, redirect, url_for, flash
from google.protobuf import text_format
from caffe.proto import caffe_pb2

import digits
from digits.config import config_option
from digits import utils
from digits.webapp import app, scheduler
from digits.dataset import ImageClassificationDatasetJob
from digits.model import tasks
from forms import ImageClassificationModelForm
from job import ImageClassificationModelJob
from digits.status import Status

NAMESPACE = '/models/images/classification'

@app.route(NAMESPACE + '/new', methods=['GET'])
def image_classification_model_new():
    form = ImageClassificationModelForm()
    form.dataset.choices = get_datasets()
    form.standard_networks.choices = get_standard_networks()
    form.standard_networks.default = get_default_standard_network()
    form.previous_networks.choices = get_previous_networks()

    prev_network_snapshots = get_previous_network_snapshots()

    return render_template('models/images/classification/new.html', form=form, previous_network_snapshots=prev_network_snapshots, has_datasets=(len(get_datasets())==0))

@app.route(NAMESPACE, methods=['POST'])
def image_classification_model_create():
    form = ImageClassificationModelForm()
    form.dataset.choices = get_datasets()
    form.standard_networks.choices = get_standard_networks()
    form.standard_networks.default = get_default_standard_network()
    form.previous_networks.choices = get_previous_networks()

    prev_network_snapshots = get_previous_network_snapshots()

    if not form.validate_on_submit():
        return render_template('models/images/classification/new.html', form=form, previous_network_snapshots=prev_network_snapshots), 400

    datasetJob = scheduler.get_job(form.dataset.data)
    if not datasetJob:
        return 'Unknown dataset job_id "%s"' % form.dataset.data, 500

    job = None
    try:
        job = ImageClassificationModelJob(
                name        = form.model_name.data,
                dataset_id  = datasetJob.id(),
                )

        network = caffe_pb2.NetParameter()
        pretrained_model = None
        if form.method.data == 'standard':
            found = False
            networks_dir = os.path.join(os.path.dirname(digits.__file__), 'standard-networks')
            for filename in os.listdir(networks_dir):
                path = os.path.join(networks_dir, filename)
                if os.path.isfile(path):
                    match = re.match(r'%s.prototxt' % form.standard_networks.data, filename)
                    if match:
                        with open(path) as infile:
                            text_format.Merge(infile.read(), network)
                        found = True
                        break
            if not found:
                raise Exception('Unknown standard model "%s"' % form.standard_networks.data)
        elif form.method.data == 'previous':
            old_job = scheduler.get_job(form.previous_networks.data)
            if not old_job:
                raise Exception('Job not found: %s' % form.previous_networks.data)
            network.CopyFrom(old_job.train_task().network)
            for i, choice in enumerate(form.previous_networks.choices):
                if choice[0] == form.previous_networks.data:
                    epoch = int(request.form['%s-snapshot' % form.previous_networks.data])
                    if epoch != 0:
                        for filename, e in old_job.train_task().snapshots:
                            if e == epoch:
                                pretrained_model = filename
                                break

                        if pretrained_model is None:
                            raise Exception("For the job %s, selected pretrained_model for epoch %d is invalid!" % (form.previous_networks.data, epoch))
                        if not (os.path.exists(pretrained_model)):
                            raise Exception("Pretrained_model for the selected epoch doesn't exists. May be deleted by another user/process. Please restart the server to load the correct pretrained_model details")
                    break

        elif form.method.data == 'custom':
            text_format.Merge(form.custom_network.data, network)
            pretrained_model = form.custom_network_snapshot.data.strip()
        else:
            raise Exception('Unrecognized method: "%s"' % form.method.data)

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
            return 'Invalid policy', 404

        job.tasks.append(
                tasks.CaffeTrainTask(
                    job_dir         = job.dir(),
                    dataset         = datasetJob,
                    train_epochs    = form.train_epochs.data,
                    snapshot_epochs = form.snapshot_epochs.data,
                    learning_rate   = form.learning_rate.data,
                    lr_policy       = policy,
                    batch_size      = form.batch_size.data,
                    val_interval    = form.val_interval.data,
                    pretrained_model= pretrained_model,
                    crop_size       = form.crop_size.data,
                    use_mean        = form.use_mean.data,
                    network         = network,
                    )
                )

        scheduler.add_job(job)
        return redirect(url_for('models_show', job_id=job.id()))

    except:
        if job:
            scheduler.delete_job(job)
        raise

def show(job):
    """
    Called from digits.views.show_job()
    """
    return render_template('models/images/classification/show.html', job=job)

@app.route(NAMESPACE + '/test_one', methods=['POST'])
def image_classification_model_test_one():
    job = scheduler.get_job(request.args['job_id'])
    if not job:
        abort(404)

    image = None
    if request.form['image_url']:
        image = utils.image.load_image(request.form['image_url'])
    elif request.files['image_file']:
        with tempfile.NamedTemporaryFile() as outfile:
            request.files['image_file'].save(outfile.name)
            image = utils.image.load_image(outfile.name)
    if image is None:
        return 'There was a problem with the image.', 400
    task = job.train_task().dataset.train_db_task()
    image = utils.image.resize_image(image, task.image_dims[0], task.image_dims[1],
            channels = task.image_dims[2],
            resize_mode = task.resize_mode,
            )
    epoch = int(request.form['snapshot_epoch'])
    predictions, visualizations = job.train_task().infer_one(image, snapshot_epoch=epoch, layers='all')
    # take top 5
    predictions = [(p[0], round(100.0*p[1],2)) for p in predictions[:5]]
    # embed as html
    visualizations = [(
        v[0],
        utils.image.embed_image_html(v[1]),
        utils.image.embed_image_html(v[2]),
        )
        for v in visualizations]
    return render_template('models/images/classification/infer_one.html',
            image_src       = utils.image.embed_image_html(image),
            predictions     = predictions,
            visualizations  = visualizations,
            )

@app.route(NAMESPACE + '/test_many', methods=['POST'])
def image_classification_model_test_many():
    job = scheduler.get_job(request.args['job_id'])
    if not job:
        abort(404)

    image_list = request.files['image_list']
    if not image_list:
        return 'File upload not found', 400

    epoch = int(request.form['snapshot_epoch'])
    if not request.form['top_n'].strip():
        top_n = 9
    else:
        top_n = int(request.form['top_n'])
    if not request.form['num_test_images'].strip():
        num_images = None
    else:
        num_images = int(request.form['num_test_images'])

    paths = []
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
        paths.append(path)
    random.shuffle(paths)

    images = []
    dataset = job.train_task().dataset
    for path in paths:
        image = utils.image.load_image(path)
        if image is not None:
            image = utils.image.resize_image(image,
                    dataset.image_dims[0], dataset.image_dims[1],
                    channels    = dataset.image_dims[2],
                    resize_mode = dataset.resize_mode,
                    )
            images.append(image)
            if num_images and len(images) >= num_images:
                break

    if not len(images):
        return 'Unable to load any images from the file', 400

    labels, scores = job.train_task().infer_many(images, snapshot_epoch=epoch)
    if scores is None:
        return 'An error occured while processing the images', 500

    indices = (-scores).argsort(axis=0)[:top_n]
    results = []
    for i in xrange(indices.shape[1]):
        result_images = []
        for j in xrange(top_n):
            result_images.append(images[indices[j][i]])
        results.append((
                labels[i],
                utils.image.embed_image_html(
                    utils.image.vis_square(np.array(result_images))
                    )
                ))

    return render_template('models/images/classification/infer_many.html',
            job=job,
            results=results,
            )

def get_datasets():
    return [(j.id(), j.name()) for j in sorted(
        [j for j in scheduler.jobs if isinstance(j, ImageClassificationDatasetJob) and (j.status.is_running() or j.status == Status.DONE)],
        cmp=lambda x,y: cmp(y.id(), x.id())
        )
        ]

def get_standard_networks():
    return [
            ('lenet', 'LeNet'),
            ('alexnet', 'AlexNet'),
            #('vgg-16', 'VGG (16-layer)'), #XXX model won't learn
            ]

def get_default_standard_network():
    return 'alexnet'

def get_previous_networks():
    return [(j.id(), j.name()) for j in sorted(
        [j for j in scheduler.jobs if isinstance(j, ImageClassificationModelJob)],
        cmp=lambda x,y: cmp(y.id(), x.id())
        )
        ]

def get_previous_network_snapshots():
    prev_network_snapshots = []
    for job_id, _ in get_previous_networks():
        job = scheduler.get_job(job_id)
        e = [(0, 'None')] + [(epoch, 'Epoch #%s' % epoch)
                for _, epoch in reversed(job.train_task().snapshots)]
        prev_network_snapshots.append(e)
    return prev_network_snapshots
