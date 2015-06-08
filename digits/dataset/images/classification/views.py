# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os

import flask

from digits import utils
from digits.utils.routing import request_wants_json
from digits.webapp import app, scheduler, autodoc
from digits.dataset import tasks
from forms import ImageClassificationDatasetForm
from job import ImageClassificationDatasetJob

NAMESPACE = '/datasets/images/classification'

def from_folders(job, form):
    """
    Add tasks for creating a dataset by parsing folders of images
    """
    job.labels_file = utils.constants.LABELS_FILE

    ### Add ParseFolderTask

    percent_val = form.folder_pct_val.data
    val_parents = []
    if form.has_val_folder.data:
        percent_val = 0

    percent_test = form.folder_pct_test.data
    test_parents = []
    if form.has_test_folder.data:
        percent_test = 0

    parse_train_task = tasks.ParseFolderTask(
            job_dir         = job.dir(),
            folder          = form.folder_train.data,
            percent_val     = percent_val,
            percent_test    = percent_test,
            )
    job.tasks.append(parse_train_task)

    # set parents
    if not form.has_val_folder.data:
        val_parents = [parse_train_task]
    if not form.has_test_folder.data:
        test_parents = [parse_train_task]

    if form.has_val_folder.data:
        parse_val_task = tasks.ParseFolderTask(
                job_dir         = job.dir(),
                parents         = parse_train_task,
                folder          = form.folder_val.data,
                percent_val     = 100,
                percent_test    = 0,
                )
        job.tasks.append(parse_val_task)
        val_parents = [parse_val_task]

    if form.has_test_folder.data:
        parse_test_task = tasks.ParseFolderTask(
                job_dir         = job.dir(),
                parents         = parse_train_task,
                folder          = form.folder_test.data,
                percent_val     = 0,
                percent_test    = 100,
                )
        job.tasks.append(parse_test_task)
        test_parents = [parse_test_task]

    ### Add CreateDbTasks

    encoding = form.encoding.data

    job.tasks.append(
            tasks.CreateDbTask(
                job_dir     = job.dir(),
                parents     = parse_train_task,
                input_file  = utils.constants.TRAIN_FILE,
                db_name     = utils.constants.TRAIN_DB,
                image_dims  = job.image_dims,
                resize_mode = job.resize_mode,
                encoding    = encoding,
                mean_file   = utils.constants.MEAN_FILE_CAFFE,
                labels_file = job.labels_file,
                )
            )

    if percent_val > 0 or form.has_val_folder.data:
        job.tasks.append(
                tasks.CreateDbTask(
                    job_dir     = job.dir(),
                    parents     = val_parents,
                    input_file  = utils.constants.VAL_FILE,
                    db_name     = utils.constants.VAL_DB,
                    image_dims  = job.image_dims,
                    resize_mode = job.resize_mode,
                    encoding    = encoding,
                    labels_file = job.labels_file,
                    )
                )

    if percent_test > 0 or form.has_test_folder.data:
        job.tasks.append(
                tasks.CreateDbTask(
                    job_dir     = job.dir(),
                    parents     = test_parents,
                    input_file  = utils.constants.TEST_FILE,
                    db_name     = utils.constants.TEST_DB,
                    image_dims  = job.image_dims,
                    resize_mode = job.resize_mode,
                    encoding    = encoding,
                    labels_file = job.labels_file,
                    )
                )

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

    image_folder = form.textfile_train_folder.data.strip()
    if not image_folder:
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

    ### val

    if form.textfile_use_val.data:
        flask.request.files[form.textfile_val_images.name].save(
                os.path.join(job.dir(), utils.constants.VAL_FILE)
                )

        image_folder = form.textfile_val_folder.data.strip()
        if not image_folder:
            image_folder = None

        job.tasks.append(
                tasks.CreateDbTask(
                    job_dir     = job.dir(),
                    input_file  = utils.constants.VAL_FILE,
                    db_name     = utils.constants.VAL_DB,
                    image_dims  = job.image_dims,
                    image_folder= image_folder,
                    resize_mode = job.resize_mode,
                    encoding    = encoding,
                    labels_file = job.labels_file,
                    shuffle     = shuffle,
                    )
                )

    ### test

    if form.textfile_use_test.data:
        flask.request.files[form.textfile_test_images.name].save(
                os.path.join(job.dir(), utils.constants.TEST_FILE)
                )

        image_folder = form.textfile_test_folder.data.strip()
        if not image_folder:
            image_folder = None

        job.tasks.append(
                tasks.CreateDbTask(
                    job_dir     = job.dir(),
                    input_file  = utils.constants.TEST_FILE,
                    db_name     = utils.constants.TEST_DB,
                    image_dims  = job.image_dims,
                    image_folder= image_folder,
                    resize_mode = job.resize_mode,
                    encoding    = encoding,
                    labels_file = job.labels_file,
                    shuffle     = shuffle,
                    )
                )


@app.route(NAMESPACE + '/new', methods=['GET'])
@autodoc('datasets')
def image_classification_dataset_new():
    """
    Returns a form for a new ImageClassificationDatasetJob
    """
    form = ImageClassificationDatasetForm()
    return flask.render_template('datasets/images/classification/new.html', form=form)

@app.route(NAMESPACE + '.json', methods=['POST'])
@app.route(NAMESPACE, methods=['POST'])
@autodoc(['datasets', 'api'])
def image_classification_dataset_create():
    """
    Creates a new ImageClassificationDatasetJob

    Returns JSON when requested: {job_id,name,status} or {errors:[]}
    """
    form = ImageClassificationDatasetForm()
    if not form.validate_on_submit():
        if request_wants_json():
            return flask.jsonify({'errors': form.errors}), 400
        else:
            return flask.render_template('datasets/images/classification/new.html', form=form), 400

    job = None
    try:
        job = ImageClassificationDatasetJob(
                name        = form.dataset_name.data,
                image_dims  = (
                    int(form.resize_height.data),
                    int(form.resize_width.data),
                    int(form.resize_channels.data),
                    ),
                resize_mode = form.resize_mode.data
                )

        if form.method.data == 'folder':
            from_folders(job, form)

        elif form.method.data == 'textfile':
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
    return flask.render_template('datasets/images/classification/show.html', job=job)

