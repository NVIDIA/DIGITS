# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import re
import tempfile
import random
import numpy as np
import scipy.io

import flask
from flask import Response
import werkzeug.exceptions
import numpy as np
from google.protobuf import text_format
try:
    import caffe_pb2
except ImportError:
    # See issue #32
    from caffe.proto import caffe_pb2

import digits
from digits.config import config_value
from digits import utils
from digits.utils.routing import request_wants_json, job_from_request
from digits.webapp import app, scheduler, autodoc
from digits.dataset import ImageClassificationDatasetJob
from digits.model import tasks
from forms import ImageClassificationModelForm
from job import ImageClassificationModelJob
from digits.status import Status
from digits.workspaces import get_workspace

NAMESPACE   = '/digits/models/images/classification'

@app.route(NAMESPACE + '/new', methods=['GET'])
@autodoc('models')
def image_classification_model_new():
    """
    Return a form for a new ImageClassificationModelJob
    """
    workspace = get_workspace(flask.request.url)
    form = ImageClassificationModelForm()
    form.dataset.choices = get_datasets()
    form.standard_networks.choices = get_standard_networks()
    form.standard_networks.default = get_default_standard_network()
    form.previous_networks.choices = get_previous_networks()

    prev_network_snapshots = get_previous_network_snapshots()

    return flask.render_template('models/images/classification/new.html',
            form = form,
            previous_network_snapshots = prev_network_snapshots,
            multi_gpu = config_value('caffe_root')['multi_gpu'],
            workspace = workspace,
            )

@app.route(NAMESPACE + '.json', methods=['POST'])
@app.route(NAMESPACE, methods=['POST'])
@autodoc(['models', 'api'])
def image_classification_model_create():
    """
    Create a new ImageClassificationModelJob

    Returns JSON when requested: {job_id,name,status} or {errors:[]}
    """
    workspace = get_workspace(flask.request.url)
    form = ImageClassificationModelForm()
    form.dataset.choices = get_datasets()
    form.standard_networks.choices = get_standard_networks()
    form.standard_networks.default = get_default_standard_network()
    form.previous_networks.choices = get_previous_networks()

    prev_network_snapshots = get_previous_network_snapshots()

    if not form.validate_on_submit():
        if request_wants_json():
            return flask.jsonify({'errors': form.errors}), 400
        else:
            return flask.render_template('models/images/classification/new.html',
                    form = form,
                    previous_network_snapshots = prev_network_snapshots,
                    multi_gpu = config_value('caffe_root')['multi_gpu'],
                    workspace = workspace,
                    ), 400

    datasetJob = scheduler.get_job(form.dataset.data)
    if not datasetJob:
        raise werkzeug.exceptions.BadRequest(
                'Unknown dataset job_id "%s"' % form.dataset.data)

    job = None
    try:
        job = ImageClassificationModelJob(
                name        = form.model_name.data,
                dataset_id  = datasetJob.id(),
                workspace = workspace,
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
                raise werkzeug.exceptions.BadRequest(
                        'Unknown standard model "%s"' % form.standard_networks.data)
        elif form.method.data == 'previous':
            old_job = scheduler.get_job(form.previous_networks.data)
            if not old_job:
                raise werkzeug.exceptions.BadRequest(
                        'Job not found: %s' % form.previous_networks.data)

            network.CopyFrom(old_job.train_task().network)
            # Rename the final layer
            # XXX making some assumptions about network architecture here
            ip_layers = [l for l in network.layer if l.type == 'InnerProduct']
            if len(ip_layers) > 0:
                ip_layers[-1].name = '%s_retrain' % ip_layers[-1].name

            for choice in form.previous_networks.choices:
                if choice[0] == form.previous_networks.data:
                    epoch = float(flask.request.form['%s-snapshot' % form.previous_networks.data])
                    if epoch != 0:
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
            text_format.Merge(form.custom_network.data, network)
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

        job.tasks.append(
                tasks.CaffeTrainTask(
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
                    use_mean        = bool(form.use_mean.data),
                    network         = network,
                    random_seed     = form.random_seed.data,
                    solver_type     = form.solver_type.data,
                    )
                )

        scheduler.add_job(job)
        if request_wants_json():
            return flask.jsonify(job.json_dict())
        else:
            return flask.redirect(flask.url_for('models_show', job_id=job.id())+'?workspace='+workspace)

    except:
        if job:
            scheduler.delete_job(job)
        raise

def show(job, *args):
    """
    Called from digits.model.views.models_show()
    """
    return flask.render_template('models/images/classification/show.html', job=job, workspace = args[0])

@app.route(NAMESPACE + '/large_graph', methods=['GET'])
@autodoc('models')
def image_classification_model_large_graph():
    """
    Show the loss/accuracy graph, but bigger
    """
    workspace = get_workspace(flask.request.url)
    job = job_from_request()
 
    return flask.render_template('models/images/classification/large_graph.html', job=job, workspace = workspace)

@app.route(NAMESPACE + '/visualize_one.json', methods=['POST'])
@app.route(NAMESPACE + '/classify_one.json', methods=['POST'])
@app.route(NAMESPACE + '/classify_one', methods=['POST', 'GET'])
@autodoc(['models', 'api'])
def image_classification_model_classify_one():
    """
    Classify one image and return the top 5 classifications

    Returns JSON when requested: {predictions: {category: confidence,...}}
    """
    job = job_from_request()
    workspace = get_workspace(flask.request.url)
    image = None
    if 'image_url' in flask.request.form and flask.request.form['image_url']:
        image = utils.image.load_image(flask.request.form['image_url'])
    elif 'image_file' in flask.request.files and flask.request.files['image_file']:
        with tempfile.NamedTemporaryFile() as outfile:
            flask.request.files['image_file'].save(outfile.name)
            image = utils.image.load_image(outfile.name)
    else:
        raise werkzeug.exceptions.BadRequest('Must provide image_url or image_file')

    # resize image
    db_task = job.train_task().dataset.train_db_task()
    height = db_task.image_dims[0]
    width = db_task.image_dims[1]
    if job.train_task().crop_size:
        height = job.train_task().crop_size
        width = job.train_task().crop_size
    image = utils.image.resize_image(image, height, width,
            channels = db_task.image_dims[2],
            resize_mode = db_task.resize_mode,
            )

    epoch = None
    if 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])

    layers = 'none'
    if 'show_visualizations' in flask.request.form and flask.request.form['show_visualizations']:
        if 'select_visualization_layer' in flask.request.form and flask.request.form['select_visualization_layer']:
            layers = flask.request.form['select_visualization_layer']
        else:
            layers = 'all'

    #vis_json = False
    #if 'visualization_json' in flask.request.form and flask.request.form['visualization_json']:
    #    vis_json = True

    save_vis_file = False
    save_file_type = ''
    save_vis_file_location = ''
    if 'save_vis_file' in flask.request.form and flask.request.form['save_vis_file']:
        save_vis_file = True
        if 'save_type_mat' in flask.request.form and flask.request.form['save_type_mat']:
            save_file_type = 'mat'
        elif 'save_type_numpy' in flask.request.form and flask.request.form['save_type_numpy']:
            save_file_type = 'numpy'
        else:
            raise werkzeug.exceptions.BadRequest('No filetype selected. Expected .npy or .mat')
        if 'save_vis_file_location' in flask.request.form and flask.request.form['save_vis_file_location']:
            save_vis_file_location = flask.request.form['save_vis_file_location']
        else:
            raise werkzeug.exceptions.BadRequest('save_vis_file_location not provided.')
    
    if 'job_id' in flask.request.form and flask.request.form['job_id']:
        job_id = flask.request.form['job_id']
    elif 'job_id' in flask.request.args and flask.request.args['job_id']:
        job_id = flask.request.args['job_id']
    else:
        raise werkzeug.exceptions.BadRequest('job_id is a necessary parameter, not found.')

    predictions, visualizations = job.train_task().infer_one(image, snapshot_epoch=epoch, layers=layers)
    # take top 5
    predictions = [(p[0], round(100.0*p[1],2)) for p in predictions[:5]]

    if save_vis_file:
        if save_file_type == 'numpy':
            try:
                np.array(visualizations).dump(open(save_vis_file_location+'/visualization_'+job_id+'.npy', 'wb'))
            except:
                raise werkzeug.exceptions.BadRequest('Error saving visualization data as Numpy array')
        elif save_file_type == 'mat':
            try:
                scipy.io.savemat(save_vis_file_location+'/visualization_'+job_id+'.mat', {'visualizations':visualizations})
            except IOError as e:
                raise werkzeug.exceptions.BadRequest('I/O error{%s}: %s'% (e.errno, e.strerror))
            except:
                raise werkzeug.exceptions.BadRequest('Error saving visualization data as .mat file')
        else:
            raise werkzeug.exceptions.BadRequest('Invalid filetype for visualization data saving')

    if request_wants_json():
        if 'show_visualizations' in flask.request.form and flask.request.form['show_visualizations']:
            # flask.jsonify has problems creating JSON from numpy.float32
            # convert all non-dict, non-list and non-string elements to string.
            for layer in visualizations:
                for ele in layer:
                    if not isinstance(layer[ele], dict) and not isinstance(layer[ele], str) and not isinstance(layer[ele], list):
                        layer[ele] = str(layer[ele]) 
            return flask.jsonify({'predictions': predictions, 'visualizations': visualizations})
        else:
            return flask.jsonify({'predictions': predictions})
    else:
        return flask.render_template('models/images/classification/classify_one.html',
                image_src       = utils.image.embed_image_html(image),
                predictions     = predictions,
                visualizations  = visualizations,
                workspace = workspace,
                )

@app.route(NAMESPACE + '/classify_many.json', methods=['POST'])
@app.route(NAMESPACE + '/classify_many', methods=['POST', 'GET'])
@autodoc(['models', 'api'])
def image_classification_model_classify_many():
    """
    Classify many images and return the top 5 classifications for each

    Returns JSON when requested: {classifications: {filename: [[category,confidence],...],...}}
    """
    job = job_from_request()
    workspace = get_workspace(flask.request.url)
    image_list = flask.request.files['image_list']
    if not image_list:
        raise werkzeug.exceptions.BadRequest('image_list is a required field')

    epoch = None
    if 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])

    paths = []
    images = []
    dataset = job.train_task().dataset

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
            image = utils.image.resize_image(image,
                    dataset.image_dims[0], dataset.image_dims[1],
                    channels    = dataset.image_dims[2],
                    resize_mode = dataset.resize_mode,
                    )
            paths.append(path)
            images.append(image)
        except utils.errors.LoadImageError as e:
            print e

    if not len(images):
        raise werkzeug.exceptions.BadRequest('Unable to load any images from the file')

    save_vis_file = False
    layers = None
    if 'save_visualizations' in flask.request.form and flask.request.form['save_visualizations']:
        save_vis_file = True

        # Check for specific layer
        if 'select_visualization_layer_bulk' in flask.request.form and flask.request.form['select_visualization_layer_bulk']:
            layers = flask.request.form['select_visualization_layer_bulk']
        else:
            layers = 'all'

        # Select save file type
        if 'save_type_mat_bulk' in flask.request.form and flask.request.form['save_type_mat_bulk']:
            save_file_type = 'mat'
        elif 'save_type_numpy_bulk' in flask.request.form and flask.request.form['save_type_numpy_bulk']:
            save_file_type = 'numpy'
        else:
            raise werkzeug.exceptions.BadRequest('No filetype selected. Expected .npy or .mat')
        
        # Obtain savefile path.
        if 'save_vis_file_location_bulk' in flask.request.form and flask.request.form['save_vis_file_location_bulk']:
            save_vis_file_location = flask.request.form['save_vis_file_location_bulk']

    if 'job_id' in flask.request.form and flask.request.form['job_id']:
        job_id = flask.request.form['job_id']
    elif 'job_id' in flask.request.args and flask.request.args['job_id']:
        job_id = flask.request.args['job_id']
    else:
        raise werkzeug.exceptions.BadRequest('job_id is a necessary parameter, not found.')

    labels, scores, visualizations = job.train_task().infer_many(images, snapshot_epoch=epoch, layers=layers)
    
    if scores is None:
        raise werkzeug.exceptions.BadRequest('An error occured while processing the images')

    # take top 5
    indices = (-scores).argsort()[:, :5]

    classifications = []
    for image_index, index_list in enumerate(indices):
        result = []
        for i in index_list:
            # `i` is a category in labels and also an index into scores
            result.append((labels[i], round(100.0*scores[image_index, i],2)))
        classifications.append(result)

    layer_data = {}
    for image_vis in visualizations:
        for layer in image_vis:
            for ele in layer:
                if ele=='image_html':
                    continue
                if layer['name'] in layer_data:
                    if ele in layer_data[layer['name']]:
                        layer_data[layer['name']][ele].append(layer[ele])
                    else:
                        layer_data[layer['name']][ele] = [layer[ele]]
                else:
                    layer_data[layer['name']] = {}
                    layer_data[layer['name']][ele] = [layer[ele]]

    if save_vis_file:
        if save_file_type == 'numpy':
            try:
                joined_vis = layer_data
                np.array(joined_vis).dump(open(save_vis_file_location+'/visualization_'+job_id+'.npy', 'wb'))
            except:
                raise werkzeug.exceptions.BadRequest('Error saving visualization data as Numpy array')
        elif save_file_type == 'mat':
            try:
                joined_vis = layer_data
                scipy.io.savemat(save_vis_file_location+'/visualization_'+job_id+'.mat', {'visualizations':joined_vis})
            except IOError as e:
                raise werkzeug.exceptions.BadRequest('I/O error{%s}: %s'% (e.errno, e.strerror))
            except:
                raise werkzeug.exceptions.BadRequest('Error saving visualization data as .mat file')
        else:
            raise werkzeug.exceptions.BadRequest('Invalid filetype for visualization data saving')

    if request_wants_json():
        if 'save_visualizations' in flask.request.form and flask.request.form['save_visualizations']:
            joined_vis = layer_data
            joined_class = dict(zip(paths, classifications))
            return flask.jsonify({'classifications': joined_class, 'visualizations': joined_vis})
        else:
            joined = dict(zip(paths, classifications))
            return flask.jsonify({'classifications': joined})
    else:
        return flask.render_template('models/images/classification/classify_many.html',
                paths=paths,
                classifications=classifications,
                workspace = workspace,
                )

@app.route(NAMESPACE + '/top_n', methods=['POST'])
@autodoc('models')
def image_classification_model_top_n():
    """
    Classify many images and show the top N images per category by confidence
    """
    job = job_from_request()
    workspace = get_workspace(flask.request.url)
    image_list = flask.request.files.get['image_list']
    if not image_list:
        raise werkzeug.exceptions.BadRequest('File upload not found')

    epoch = None
    if 'snapshot_epoch' in flask.request.form:
        epoch = float(flask.request.form['snapshot_epoch'])
    if 'top_n' in flask.request.form and flask.request.form['top_n'].strip():
        top_n = int(flask.request.form['top_n'])
    else:
        top_n = 9
    if 'num_test_images' in flask.request.form and flask.request.form['num_test_images'].strip():
        num_images = int(flask.request.form['num_test_images'])
    else:
        num_images = None

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
        try:
            image = utils.image.load_image(path)
            image = utils.image.resize_image(image,
                    dataset.image_dims[0], dataset.image_dims[1],
                    channels    = dataset.image_dims[2],
                    resize_mode = dataset.resize_mode,
                    )
            images.append(image)
            if num_images and len(images) >= num_images:
                break
        except utils.errors.LoadImageError as e:
            print e

    if not len(images):
        raise werkzeug.exceptions.BadRequest(
                'Unable to load any images from the file')

    labels, scores = job.train_task().infer_many(images, snapshot_epoch=epoch)
    if scores is None:
        raise RuntimeError('An error occured while processing the images')

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

    return flask.render_template('models/images/classification/top_n.html',
            job=job,
            results=results,
            workspace = workspace,
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
            ('googlenet', 'GoogLeNet'),
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

