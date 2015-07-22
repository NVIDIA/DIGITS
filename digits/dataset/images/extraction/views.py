# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os

import flask

from digits import utils
from digits.utils.routing import request_wants_json
from digits.webapp import app, scheduler, autodoc
from digits.dataset import tasks
from forms import FeatureExtractionDatasetForm
from job import FeatureExtractionDatasetJob

NAMESPACE = '/datasets/images/extraction'

def from_files(job, form):
    """
    Add tasks for creating a dataset by reading textfiles
    """
    ### labels

    flask.request.files[form.textfile_labels_file.name].save(
            os.path.join(job.dir(), utils.constants.LABELS_FILE)
            )
    job.labels_file = utils.constants.LABELS_FILE

    encoding = form.encoding.data
    shuffle = bool(form.textfile_shuffle.data)

    ### train

    flask.request.files[form.textfile_train_images.name].save(
            os.path.join(job.dir(), utils.constants.TRAIN_FILE)
            )

    #image_folder = form.textfile_train_folder.data.strip()
    #if not image_folder:
    image_folder = None

    job.tasks.append(
            tasks.CreateDbTask(
                job_dir     = job.dir(),
                input_file  = utils.constants.TRAIN_FILE,
                db_name     = utils.constants.TRAIN_DB,
                image_dims  = job.image_dims,
                image_folder= image_folder,
                resize_mode = job.resize_mode,
                encoding    = encoding,
                mean_file   = utils.constants.MEAN_FILE_CAFFE,
                labels_file = job.labels_file,
                shuffle     = shuffle,
                )
            )

@app.route(NAMESPACE + '/new', methods=['GET'])
@autodoc('datasets')
def feature_extraction_dataset_new():
    """
    Returns a form for a new FeatureExtractionDatasetJob
    """
    form = FeatureExtractionDatasetForm()
    return flask.render_template('datasets/images/extraction/new.html', form=form)

@app.route(NAMESPACE + '.json', methods=['POST'])
@app.route(NAMESPACE, methods=['POST'])
@autodoc(['datasets', 'api'])
def feature_extraction_dataset_create():
    """
    Creates a new FeatureExtractionDatasetJob

    Returns JSON when requested: {job_id,name,status} or {errors:[]}
    """
    form = FeatureExtractionDatasetForm()
    if not form.validate_on_submit():
        if request_wants_json():
            return flask.jsonify({'errors': form.errors}), 400
        else:
            return flask.render_template('datasets/images/extraction/new.html', form=form), 400

    job = None
    try:
        job = FeatureExtractionDatasetJob(
                name        = form.dataset_name.data,
                image_dims  = (
                    int(form.resize_height.data),
                    int(form.resize_width.data),
                    int(form.resize_channels.data),
                    ),
                resize_mode = form.resize_mode.data
                )

        #if form.method.data == 'folder':
        #    from_folders(job, form)

        #elif form.method.data == 'textfile':
        from_files(job, form)

        scheduler.add_job(job)
        if request_wants_json():
            return flask.jsonify(job.json_dict())
        else:
            return flask.redirect(flask.url_for('datasets_show', job_id=job.id()))

    except:
        if job:
            scheduler.delete_job(job)
        raise

def show(job):
    """
    Called from digits.dataset.views.datasets_show()
    """
    return flask.render_template('datasets/images/extraction/show.html', job=job)

