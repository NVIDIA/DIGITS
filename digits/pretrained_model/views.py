# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
import json
import os
import shutil
import tempfile
import tarfile
import zipfile

import flask
import h5py
import json
import numpy as np
import PIL.Image
import StringIO
import werkzeug.exceptions

from digits import dataset, extensions, model, utils
from digits.inference import WeightsJob, GradientAscentJob
from digits.pretrained_model import PretrainedModelJob

from digits.utils.routing import request_wants_json, job_from_request
from digits.views import get_job_list
from digits.webapp import app, scheduler


blueprint = flask.Blueprint(__name__, __name__)

def get_tempfile(f, suffix):
    temp = tempfile.mkstemp(suffix=suffix)
    f.save(temp[1])
    path = temp[1]
    os.close(temp[0])
    return path

def validate_caffe_files(files):
    """
    Upload a caffemodel
    """
    # Validate model weights:
    if str(files['weights_file'].filename) is '':
        raise werkzeug.exceptions.BadRequest('Missing weights file')
    elif files['weights_file'].filename.rsplit('.',1)[1] != "caffemodel" :
        raise werkzeug.exceptions.BadRequest('Weights must be a .caffemodel file')

    # Validate model definition:
    if str(files['model_def_file'].filename) is '':
        raise werkzeug.exceptions.BadRequest('Missing model definition file')
    elif files['model_def_file'].filename.rsplit('.',1)[1] != "prototxt" :
        raise werkzeug.exceptions.BadRequest('Model definition must be .prototxt file')

    weights_path   = get_tempfile(flask.request.files['weights_file'],".caffemodel")
    model_def_path = get_tempfile(flask.request.files['model_def_file'],".prototxt")

    return (weights_path, model_def_path)

def validate_torch_files(files):
    """
    Upload a torch model
    """
    # Validate model weights:
    if str(files['weights_file'].filename) is '':
        raise werkzeug.exceptions.BadRequest('Missing weights file')
    elif files['weights_file'].filename.rsplit('.',1)[1] != "t7" :
        raise werkzeug.exceptions.BadRequest('Weights must be a .t7 file')

    # Validate model definition:
    if str(files['model_def_file'].filename) is '':
        raise werkzeug.exceptions.BadRequest('Missing model definition file')
    elif files['model_def_file'].filename.rsplit('.',1)[1] != "lua" :
        raise werkzeug.exceptions.BadRequest('Model definition must be .lua file')

    weights_path   = get_tempfile(flask.request.files['weights_file'],".t7")
    model_def_path = get_tempfile(flask.request.files['model_def_file'],".lua")

    return (weights_path, model_def_path)

def validate_archive_keys(info):
    """
    Validate keys stored in the info.json file
    """
    keys = ["snapshot file", "framework", "name"]
    for key in keys:
        if key not in info:
            return (False, key)

    return (True, 0)

def format_job_name(job):
    return {"name": job.name(), "id": job.id()}

def create_max_activation_job(job,username,layer_name,units):
    gradient_ascent_job = GradientAscentJob(
        job,
        layer_name,
        units,
        name = "Gradient Ascent: %s, %s" % (job.name(), layer_name),
        username = username
    )
    scheduler.add_job(gradient_ascent_job)
    return gradient_ascent_job

@utils.auth.requires_login
@blueprint.route('/run_max_activations.json', methods=['POST'])
def run_max_activations():
    """ Run Gradient Ascent on a given layer and units """
    job  = scheduler.get_job(flask.request.args["job_id"])
    args = flask.request.args
    layer_name = args["layer_name"]
    units = eval(args["units"])
    username = utils.auth.get_username()

    gradient_ascent_job = create_max_activation_job(job,username,layer_name,units)

    return flask.jsonify({"stats": units, "job_id": gradient_ascent_job.id()})

def fill_empty(num):
    data = []
    for unit in range(num):
        data.append([[[]]])
    return data

def serve_pil_image(pil_img):
    img_io = StringIO.StringIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return flask.send_file(img_io, mimetype='image/jpeg')

@blueprint.route('/max_activation', methods=['GET'])
def max_activation():

    args = flask.request.args
    job = job_from_request()
    layer_name = args["layer_name"]
    unit       = args["unit"]

    raw_data = 128*np.ones((256,256)).astype(int)

    max_activation_path = job.get_max_activations_path()
    if os.path.isfile(max_activation_path):
        f = h5py.File(max_activation_path,'r')
        if layer_name in f:
            if str(unit) in f[layer_name]:
                raw_data = np.transpose(f[layer_name][str(unit)]['cropped'][:],(1,2,0))
        f.close()
    # Add one channel for greyscale images:
    if len(raw_data[0][0]) == 1:
        raw_data = np.transpose(raw_data, (2,0,1))[0]

    img = PIL.Image.fromarray(np.uint8(raw_data))
    return serve_pil_image(img)

@blueprint.route('/stop_max_activations.json', methods=['GET'])
def stop_max_activations():
    """ Stop Max Activations Job """
    job  = job_from_request()
    args = flask.request.args
    job_id = args["gradient_ascent_id"]
    scheduler.abort_job(job_id)
    return flask.jsonify({"status": "success"}), 200

@blueprint.route('/remove_max_activations.json', methods=['GET'])
def remove_max_activations():
    """ Deletes Max Activations Dataset pertaining to specified layer in job """
    job = job_from_request()
    args = flask.request.args
    layer_name = args["layer_name"]

    if os.path.isfile(job.get_max_activations_path()):
        f = h5py.File(job.get_max_activations_path(),'a')
        if layer_name in f:
            del f[layer_name]
            return flask.jsonify({"status": "success"}), 200

    return flask.jsonify({"status": "error"}), 500

def layer_has_max_activations(job,layer_name):
    if os.path.isfile(job.get_max_activations_path()):
        f = h5py.File(job.get_max_activations_path(),'r')
        if layer_name in f:
            return True
    return False

@blueprint.route('/get_max_activations.json', methods=['GET'])
def get_max_activations():
    """ Returns array of maximum activations for a given layer """
    job = job_from_request()
    args = flask.request.args
    layer_name = args["layer_name"]
    range_min  = int(args["range_min"])
    range_max  = int(args["range_max"])
    data = []
    stats = {}

    if layer_has_max_activations(job,layer_name):
        f = h5py.File(job.get_max_activations_path(),'r')

        completed_units = len(f[layer_name].keys())
        for unit in range(completed_units):
            data.append(True)

        w = h5py.File(job.get_filters_path(),'r')
        if layer_name in w:
            stats = json.loads(w[layer_name].attrs["stats"])
            total_units = stats["shape"][0]
            uncompleted_units = total_units - completed_units

            if uncompleted_units > 0:
                for unit in range(uncompleted_units):
                    data.append(False)

    elif os.path.isfile(job.get_filters_path()):
        f = h5py.File(job.get_filters_path(),'r')
        if layer_name in f:
            stats = json.loads(f[layer_name].attrs["stats"])
            data = fill_empty(stats["shape"][0])

    return flask.jsonify({"stats": stats, "data": data[range_min:range_max], "length": len(data)})

@blueprint.route('/get_weights.json', methods=['GET'])
def get_weights():
    """ Return the weights for a given layer """
    job = job_from_request()

    args = flask.request.args
    layer_name = args["layer_name"]
    range_min  = int(args["range_min"])
    range_max  = int(args["range_max"])
    data   = []
    stats  = {}
    num_units = 0

    # Open h5py file, and retrieve weights in specified range for given layer:
    if os.path.isfile(job.get_filters_path()):
        f = h5py.File(job.get_filters_path())
        if layer_name in f:
            num_units = len(f[layer_name])
            stats = json.loads(f[layer_name].attrs["stats"])
            data = f[layer_name][:][range_min:range_max].tolist()

    return flask.jsonify({"data": data, "length": num_units, "stats": stats })

@blueprint.route('/get_outputs.json', methods=['GET'])
def get_outputs():
    job  = scheduler.get_job(flask.request.args["job_id"])

    # If older job, then create weights db file:
    if not job.has_weights():
        job.tasks[0].write_deploy()
        weights_job = run_weights_job(job,utils.auth.get_username())
        weights_job.wait_completion()
        # If failed to create weights, recommend re-uploading:
        status = weights_job.status.name
        if status is "Error":
            return flask.jsonify({"stats": status, "msg": "Could not generate weights, consider re-uploading job." })

    layers_with_outputs = []
    if os.path.isfile(job.get_filters_path()):
        f = h5py.File(job.get_filters_path(),'r')
        layers_with_outputs = f.keys()

    return flask.jsonify({"model_def": job.get_model_def(True), "framework": job.framework, "layers_with_outputs": layers_with_outputs})

@utils.auth.requires_login
@blueprint.route('/layer_visualizations/<job_id>', methods=['GET'])
def layer_visualizations(job_id):
    job = scheduler.get_job(job_id)
    if not job.has_deploy():
        job.tasks[0].write_deploy()

    return flask.render_template("pretrained_models/layer_visualizations.html",job=format_job_name(job))

def run_weights_job(job,username):
    """
    Run Job To Retrieve Weights From Pretrained Model
    """
    # Get Weights:
    weights_job = WeightsJob(
        job,
        name     = job.name() + " (getting weights)",
        username = username
    )

    scheduler.add_job(weights_job)
    return weights_job

@utils.auth.requires_login
@blueprint.route('/upload_archive', methods=['POST'])
def upload_archive():
    """
    Upload archive
    """
    files = flask.request.files
    archive_file = get_tempfile(files["archive"],".archive");
    labels_file  = None
    mean_file    = None

    if tarfile.is_tarfile(archive_file):
        archive = tarfile.open(archive_file,'r')
        names = archive.getnames()
    elif zipfile.is_zipfile(archive_file):
        archive = zipfile.ZipFile(archive_file, 'r')
        names = archive.namelist()
    else:
        return flask.jsonify({"status": "Incorrect Archive Type"}), 500

    if "info.json" in names:

        # Create a temp directory to storce archive
        tempdir = tempfile.mkdtemp()
        labels_file = None
        archive.extractall(path=tempdir)

        with open(os.path.join(tempdir, "info.json")) as data_file:
            info = json.load(data_file)

        valid, key = validate_archive_keys(info)

        if valid is False:
            return flask.jsonify({"status": "Missing Key '"+ key +"' in info.json"}), 500

        # Get path to files needed to be uploaded in directory
        weights_file = os.path.join(tempdir, info["snapshot file"])

        if "model file" in info:
            model_file   = os.path.join(tempdir, info["model file"])
        elif "network file" in info:
            model_file   = os.path.join(tempdir, info["network file"])
        else:
            return flask.jsonify({"status": "Missing model definition in info.json"}), 500

        if "labels file" in info:
            labels_file  = os.path.join(tempdir, info["labels file"])
        if "mean file" in info:
            mean_file  = os.path.join(tempdir, info["mean file"])

        # Upload the Model:
        job = PretrainedModelJob(
            weights_file,
            model_file ,
            labels_file,
            mean_file,
            info["framework"],
            info["image dimensions"][2],
            info["image resize mode"],
            info["image dimensions"][0],
            info["image dimensions"][1],
            username = utils.auth.get_username(),
            name = info["name"]
        )

        scheduler.add_job(job)

        run_weights_job(job, utils.auth.get_username())

        # Delete temp directory
        shutil.rmtree(tempdir, ignore_errors=True)

        return flask.jsonify({"status": "success"}), 200
    else:
        return flask.jsonify({"status": "Missing or Incorrect json file"}), 500


@utils.auth.requires_login
@blueprint.route('/new', methods=['POST'])
def new():
    """
    Upload a pretrained model
    """
    labels_path = None
    mean_path   = None
    framework   = None

    form  = flask.request.form
    files = flask.request.files

    if 'framework' not in form:
        framework = "caffe"
    else:
        framework = form['framework']

    if 'job_name' not in flask.request.form:
        raise werkzeug.exceptions.BadRequest('Missing job name')
    elif str(flask.request.form['job_name']) is '':
        raise werkzeug.exceptions.BadRequest('Missing job name')

    if framework == "caffe":
        weights_path, model_def_path = validate_caffe_files(files)
    else:
        weights_path, model_def_path = validate_torch_files(files)

    if str(flask.request.files['labels_file'].filename) is not '':
        labels_path = get_tempfile(flask.request.files['labels_file'],".txt")

    if str(flask.request.files['mean_file'].filename) is not '':
        mean_path = get_tempfile(flask.request.files['mean_file'],".prototxt")

    job = PretrainedModelJob(
        weights_path,
        model_def_path,
        labels_path,
        mean_path,
        framework,
        form["image_type"],
        form["resize_mode"],
        form["height"],
        form["width"],
        username = utils.auth.get_username(),
        name = flask.request.form['job_name']
    )
    scheduler.add_job(job)

    job.wait_completion()

    weights_job = WeightsJob(
        job,
        name     = flask.request.form['job_name'],
        username = utils.auth.get_username()
    )
    scheduler.add_job(weights_job)

    return flask.redirect(flask.url_for('digits.views.home', tab=3)), 302
