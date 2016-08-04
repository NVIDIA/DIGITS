import flask
import tempfile
import tarfile
import zipfile
import os
import shutil
import h5py
import json
import numpy as np
import PIL.Image
import StringIO
from digits import dataset, extensions, model, utils
from digits.webapp import app, scheduler
from digits.pretrained_model import PretrainedModelJob

from digits.inference import WeightsJob
from digits.inference import GradientAscentJob

from digits.utils.routing import request_wants_json, job_from_request
from digits.views import get_job_list

from digits import utils
import werkzeug.exceptions

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

def format_job_name(job):
    return {"name": job.name(), "id": job.id()}

@utils.auth.requires_login
@blueprint.route('/run_max_activations.json', methods=['POST'])
def run_max_activations():
    """ Run Gradient Ascent on a given layer and units """
    job  = scheduler.get_job(flask.request.args["job_id"])
    args = flask.request.args
    layer_name = args["layer_name"]
    units = eval(args["units"])

    gradient_ascent_job = GradientAscentJob(
        job,
        layer_name,
        units,
        name = "Gradient Ascent",
        username = utils.auth.get_username()
    )

    scheduler.add_job(gradient_ascent_job)
    gradient_ascent_job.wait_completion()
    scheduler.delete_job(gradient_ascent_job)

    # f = h5py.File(model_job.get_activations_path())
    # image_id = str(len(f.keys())-1)
    # input_data = f[image_id]['data'][:][0]

    return flask.jsonify({"stats": units})

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
    # job  = format_job_name(scheduler.get_job(job_id))
    job = job_from_request()
    layer_name = args["layer_name"]
    unit       = args["unit"]

    raw_data = 128*np.ones((256,256))

    max_activation_path = job.get_max_activations_path()
    if os.path.isfile(max_activation_path):
        f = h5py.File(max_activation_path,'r')
        if layer_name in f:
            if str(unit) in f[layer_name]:
                raw_data = f[layer_name][str(unit)]['cropped'][:]
        f.close()
    img = PIL.Image.fromarray(np.uint8(raw_data))
    return serve_pil_image(img)

@blueprint.route('/get_max_activations.json', methods=['GET'])
def get_max_activations():
    """ Returns array of maximum activations for a given layer """
    job = job_from_request()
    args = flask.request.args
    layer_name = args["layer_name"]
    data = []
    # For now return an empty array (currently just building selection)
    if os.path.isfile(job.get_max_activations_path()):
        f = h5py.File(job.get_max_activations_path(),'a')
        w = h5py.File(job.get_filters_path(),'r')
        stats = json.loads(w[layer_name].attrs["stats"])

        if layer_name in w:
            for unit in range(stats["num_activations"]):
                if layer_name in f:
                    if str(unit) in f[layer_name]:
                        data.append([[[]]])
                    else:
                        data.append([[[]]])
                else:
                    data.append([[[]]])
    elif os.path.isfile(job.get_filters_path()):
        f = h5py.File(job.get_filters_path(),'r')
        stats = json.loads(f[layer_name].attrs["stats"])
        data = fill_empty(stats["num_activations"])

    return flask.jsonify({"stats": stats, "data": data})

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
    data = []

    return flask.jsonify({"model_def": job.get_model_def(True), "images": data, "framework": job.framework})

@utils.auth.requires_login
@blueprint.route('/layer_visualizations/<job_id>', methods=['GET'])
def layer_visualizations(job_id):
    job  = format_job_name(scheduler.get_job(job_id))
    return flask.render_template("pretrained_models/layer_visualizations.html",job=job)

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
        return flask.jsonify({"status": "error"}), 500

    if "info.json" in names:

        # Create a temp directory to storce archive
        tempdir = tempfile.mkdtemp()
        archive.extractall(path=tempdir)

        with open(os.path.join(tempdir, "info.json")) as data_file:
            info = json.load(data_file)

        # Get path to files needed to be uploaded in directory
        weights_file = os.path.join(tempdir, info["snapshot file"])
        model_file   = os.path.join(tempdir, info["model file"])
        labels_file  = os.path.join(tempdir, info["labels file"])
        mean_file    = os.path.join(tempdir, info["mean file"])

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
        job.wait_completion()

        # Delete temp directory
        shutil.rmtree(tempdir, ignore_errors=True)

        # Get Weights:
        weights_job = WeightsJob(
            job,
            name     = info['name'],
            username = utils.auth.get_username()
        )

        scheduler.add_job(weights_job)

        return flask.jsonify({"status": "success"}), 200
    else:
        return flask.jsonify({"status": "error"}), 500


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
        form["width"],
        form["height"],
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

    return flask.redirect(flask.url_for('digits.views.home')), 302
