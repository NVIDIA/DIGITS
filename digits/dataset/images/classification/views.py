# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os

import flask

from digits import utils
from digits.utils.routing import request_wants_json, job_from_request
from digits.webapp import app, scheduler, autodoc
from digits.dataset import tasks
from forms import ImageClassificationDatasetForm
from job import ImageClassificationDatasetJob

NAMESPACE = '/datasets/images/classification'


def augmentation_from_form(form, job):
    augmentation = {}

    if form.has_augmentation_contrast.data:
        augmentation['contrast'] = {
            'strength_min': form.augmentation_contrast_strength_min.data,
            'strength_max': form.augmentation_contrast_strength_max.data,
            'probability': form.augmentation_contrast_probability.data
        }

    if form.has_augmentation_hue.data and job.image_dims[2] == 3:
        augmentation['hue'] = {
            'angle_min': form.augmentation_hue_angle_min.data,
            'angle_max': form.augmentation_hue_angle_max.data,
            'probability': form.augmentation_hue_probability.data
        }

    if form.has_augmentation_rotation.data:
        augmentation['rotation'] = {
            'angle_min': form.augmentation_rotation_angle_min.data,
            'angle_max': form.augmentation_rotation_angle_max.data,
            'probability': form.augmentation_rotation_probability.data
        }

    if form.has_augmentation_translation.data:
        augmentation['translation'] = {
            'dx_min': form.augmentation_translation_dx_min.data,
            'dx_max': form.augmentation_translation_dx_max.data,
            'dy_min': form.augmentation_translation_dy_min.data,
            'dy_max': form.augmentation_translation_dy_max.data,
            'probability': form.augmentation_translation_probability.data
        }
    return augmentation


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

    min_per_class = form.folder_train_min_per_class.data
    max_per_class = form.folder_train_max_per_class.data

    parse_train_task = tasks.ParseFolderTask(
            job_dir          = job.dir(),
            folder           = form.folder_train.data,
            percent_val      = percent_val,
            percent_test     = percent_test,
            min_per_category = min_per_class if min_per_class>0 else 1,
            max_per_category = max_per_class if max_per_class>0 else None
            )
    job.tasks.append(parse_train_task)

    # set parents
    if not form.has_val_folder.data:
        val_parents = [parse_train_task]
    if not form.has_test_folder.data:
        test_parents = [parse_train_task]

    if form.has_val_folder.data:
        min_per_class = form.folder_val_min_per_class.data
        max_per_class = form.folder_val_max_per_class.data

        parse_val_task = tasks.ParseFolderTask(
                job_dir         = job.dir(),
                parents         = parse_train_task,
                folder          = form.folder_val.data,
                percent_val     = 100,
                percent_test    = 0,
                min_per_category = min_per_class if min_per_class>0 else 1,
                max_per_category = max_per_class if max_per_class>0 else None
                )
        job.tasks.append(parse_val_task)
        val_parents = [parse_val_task]

    if form.has_test_folder.data:
        min_per_class = form.folder_test_min_per_class.data
        max_per_class = form.folder_test_max_per_class.data

        parse_test_task = tasks.ParseFolderTask(
                job_dir         = job.dir(),
                parents         = parse_train_task,
                folder          = form.folder_test.data,
                percent_val     = 0,
                percent_test    = 100,
                min_per_category = min_per_class if min_per_class>0 else 1,
                max_per_category = max_per_class if max_per_class>0 else None
                )
        job.tasks.append(parse_test_task)
        test_parents = [parse_test_task]

    ### Add CreateDbTasks

    backend = form.backend.data
    encoding = form.encoding.data
    compression = form.compression.data

    augmentation = augmentation_from_form(form, job)

    job.tasks.append(
            tasks.CreateDbTask(
                job_dir      = job.dir(),
                parents      = parse_train_task,
                input_file   = utils.constants.TRAIN_FILE,
                db_name      = utils.constants.TRAIN_DB,
                backend      = backend,
                image_dims   = job.image_dims,
                resize_mode  = job.resize_mode,
                encoding     = encoding,
                compression  = compression,
                mean_file    = utils.constants.MEAN_FILE_CAFFE,
                labels_file  = job.labels_file,
                augmentation = augmentation,
                )
            )

    if percent_val > 0 or form.has_val_folder.data:
        job.tasks.append(
                tasks.CreateDbTask(
                    job_dir     = job.dir(),
                    parents     = val_parents,
                    input_file  = utils.constants.VAL_FILE,
                    db_name     = utils.constants.VAL_DB,
                    backend     = backend,
                    image_dims  = job.image_dims,
                    resize_mode = job.resize_mode,
                    encoding    = encoding,
                    compression = compression,
                    labels_file = job.labels_file,
                    augmentation= {},
                    )
                )

    if percent_test > 0 or form.has_test_folder.data:
        job.tasks.append(
                tasks.CreateDbTask(
                    job_dir     = job.dir(),
                    parents     = test_parents,
                    input_file  = utils.constants.TEST_FILE,
                    db_name     = utils.constants.TEST_DB,
                    backend     = backend,
                    image_dims  = job.image_dims,
                    resize_mode = job.resize_mode,
                    encoding    = encoding,
                    compression = compression,
                    labels_file = job.labels_file,
                    augmentation= {},
                    )
                )

def from_files(job, form):
    """
    Add tasks for creating a dataset by reading textfiles
    """
    ### labels
    if form.textfile_use_local_files.data:
        job.labels_file = form.textfile_local_labels_file.data.strip()
    else:
        flask.request.files[form.textfile_labels_file.name].save(
                os.path.join(job.dir(), utils.constants.LABELS_FILE)
                )
        job.labels_file = utils.constants.LABELS_FILE

    shuffle = bool(form.textfile_shuffle.data)
    backend = form.backend.data
    encoding = form.encoding.data
    compression = form.compression.data

    ### train
    if form.textfile_use_local_files.data:
        train_file = form.textfile_local_train_images.data.strip()
    else:
        flask.request.files[form.textfile_train_images.name].save(
                os.path.join(job.dir(), utils.constants.TRAIN_FILE)
                )
        train_file = utils.constants.TRAIN_FILE

    image_folder = form.textfile_train_folder.data.strip()
    if not image_folder:
        image_folder = None

    augmentation = augmentation_from_form(form, job)

    job.tasks.append(
            tasks.CreateDbTask(
                job_dir     = job.dir(),
                input_file  = train_file,
                db_name     = utils.constants.TRAIN_DB,
                backend     = backend,
                image_dims  = job.image_dims,
                image_folder= image_folder,
                resize_mode = job.resize_mode,
                encoding    = encoding,
                compression = compression,
                mean_file   = utils.constants.MEAN_FILE_CAFFE,
                labels_file = job.labels_file,
                shuffle     = shuffle,
                augmentation= augmentation,
                )
            )

    ### val

    if form.textfile_use_val.data:
        if form.textfile_use_local_files.data:
            val_file = form.textfile_local_val_images.data.strip()
        else:
            flask.request.files[form.textfile_val_images.name].save(
                    os.path.join(job.dir(), utils.constants.VAL_FILE)
                    )
            val_file = utils.constants.VAL_FILE

        image_folder = form.textfile_val_folder.data.strip()
        if not image_folder:
            image_folder = None

        job.tasks.append(
                tasks.CreateDbTask(
                    job_dir     = job.dir(),
                    input_file  = val_file,
                    db_name     = utils.constants.VAL_DB,
                    backend     = backend,
                    image_dims  = job.image_dims,
                    image_folder= image_folder,
                    resize_mode = job.resize_mode,
                    encoding    = encoding,
                    compression = compression,
                    labels_file = job.labels_file,
                    shuffle     = shuffle,
                    augmentation= {},
                    )
                )

    ### test

    if form.textfile_use_test.data:
        if form.textfile_use_local_files.data:
            test_file = form.textfile_local_test_images.data.strip()
        else:
            flask.request.files[form.textfile_test_images.name].save(
                    os.path.join(job.dir(), utils.constants.TEST_FILE)
                    )
            test_file = utils.constants.TEST_FILE

        image_folder = form.textfile_test_folder.data.strip()
        if not image_folder:
            image_folder = None

        job.tasks.append(
                tasks.CreateDbTask(
                    job_dir     = job.dir(),
                    input_file  = test_file,
                    db_name     = utils.constants.TEST_DB,
                    backend     = backend,
                    image_dims  = job.image_dims,
                    image_folder= image_folder,
                    resize_mode = job.resize_mode,
                    encoding    = encoding,
                    compression = compression,
                    labels_file = job.labels_file,
                    shuffle     = shuffle,
                    augmentation= {},
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

        else:
            raise ValueError('method not supported')

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

@app.route(NAMESPACE + '/summary', methods=['GET'])
@autodoc('datasets')
def image_classification_dataset_summary():
    """
    Return a short HTML summary of a DatasetJob
    """
    job = job_from_request()

    return flask.render_template('datasets/images/classification/summary.html', dataset=job)

