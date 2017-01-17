# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

import caffe_pb2
import flask
import PIL.Image

from .forms import ImageClassificationDatasetForm
from .job import ImageClassificationDatasetJob
from digits import utils
from digits.dataset import tasks
from digits.utils.forms import fill_form_if_cloned, save_form_to_job
from digits.utils.lmdbreader import DbReader
from digits.utils.routing import request_wants_json, job_from_request
from digits.webapp import scheduler


blueprint = flask.Blueprint(__name__, __name__)


def from_folders(job, form):
    """
    Add tasks for creating a dataset by parsing folders of images
    """
    job.labels_file = utils.constants.LABELS_FILE

    # Add ParseFolderTask

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
        job_dir=job.dir(),
        folder=form.folder_train.data,
        percent_val=percent_val,
        percent_test=percent_test,
        min_per_category=min_per_class if min_per_class > 0 else 1,
        max_per_category=max_per_class if max_per_class > 0 else None
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
            job_dir=job.dir(),
            parents=parse_train_task,
            folder=form.folder_val.data,
            percent_val=100,
            percent_test=0,
            min_per_category=min_per_class if min_per_class > 0 else 1,
            max_per_category=max_per_class if max_per_class > 0 else None
        )
        job.tasks.append(parse_val_task)
        val_parents = [parse_val_task]

    if form.has_test_folder.data:
        min_per_class = form.folder_test_min_per_class.data
        max_per_class = form.folder_test_max_per_class.data

        parse_test_task = tasks.ParseFolderTask(
            job_dir=job.dir(),
            parents=parse_train_task,
            folder=form.folder_test.data,
            percent_val=0,
            percent_test=100,
            min_per_category=min_per_class if min_per_class > 0 else 1,
            max_per_category=max_per_class if max_per_class > 0 else None
        )
        job.tasks.append(parse_test_task)
        test_parents = [parse_test_task]

    # Add CreateDbTasks

    backend = form.backend.data
    encoding = form.encoding.data
    compression = form.compression.data

    job.tasks.append(
        tasks.CreateDbTask(
            job_dir=job.dir(),
            parents=parse_train_task,
            input_file=utils.constants.TRAIN_FILE,
            db_name=utils.constants.TRAIN_DB,
            backend=backend,
            image_dims=job.image_dims,
            resize_mode=job.resize_mode,
            encoding=encoding,
            compression=compression,
            mean_file=utils.constants.MEAN_FILE_CAFFE,
            labels_file=job.labels_file,
        )
    )

    if percent_val > 0 or form.has_val_folder.data:
        job.tasks.append(
            tasks.CreateDbTask(
                job_dir=job.dir(),
                parents=val_parents,
                input_file=utils.constants.VAL_FILE,
                db_name=utils.constants.VAL_DB,
                backend=backend,
                image_dims=job.image_dims,
                resize_mode=job.resize_mode,
                encoding=encoding,
                compression=compression,
                labels_file=job.labels_file,
            )
        )

    if percent_test > 0 or form.has_test_folder.data:
        job.tasks.append(
            tasks.CreateDbTask(
                job_dir=job.dir(),
                parents=test_parents,
                input_file=utils.constants.TEST_FILE,
                db_name=utils.constants.TEST_DB,
                backend=backend,
                image_dims=job.image_dims,
                resize_mode=job.resize_mode,
                encoding=encoding,
                compression=compression,
                labels_file=job.labels_file,
            )
        )


def from_files(job, form):
    """
    Add tasks for creating a dataset by reading textfiles
    """
    # labels
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

    # train
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

    job.tasks.append(
        tasks.CreateDbTask(
            job_dir=job.dir(),
            input_file=train_file,
            db_name=utils.constants.TRAIN_DB,
            backend=backend,
            image_dims=job.image_dims,
            image_folder=image_folder,
            resize_mode=job.resize_mode,
            encoding=encoding,
            compression=compression,
            mean_file=utils.constants.MEAN_FILE_CAFFE,
            labels_file=job.labels_file,
            shuffle=shuffle,
        )
    )

    # val

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
                job_dir=job.dir(),
                input_file=val_file,
                db_name=utils.constants.VAL_DB,
                backend=backend,
                image_dims=job.image_dims,
                image_folder=image_folder,
                resize_mode=job.resize_mode,
                encoding=encoding,
                compression=compression,
                labels_file=job.labels_file,
                shuffle=shuffle,
            )
        )

    # test

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
                job_dir=job.dir(),
                input_file=test_file,
                db_name=utils.constants.TEST_DB,
                backend=backend,
                image_dims=job.image_dims,
                image_folder=image_folder,
                resize_mode=job.resize_mode,
                encoding=encoding,
                compression=compression,
                labels_file=job.labels_file,
                shuffle=shuffle,
            )
        )


@blueprint.route('/new', methods=['GET'])
@utils.auth.requires_login
def new():
    """
    Returns a form for a new ImageClassificationDatasetJob
    """
    form = ImageClassificationDatasetForm()

    # Is there a request to clone a job with ?clone=<job_id>
    fill_form_if_cloned(form)

    return flask.render_template('datasets/images/classification/new.html', form=form)


@blueprint.route('.json', methods=['POST'])
@blueprint.route('', methods=['POST'], strict_slashes=False)
@utils.auth.requires_login(redirect=False)
def create():
    """
    Creates a new ImageClassificationDatasetJob

    Returns JSON when requested: {job_id,name,status} or {errors:[]}
    """
    form = ImageClassificationDatasetForm()

    # Is there a request to clone a job with ?clone=<job_id>
    fill_form_if_cloned(form)

    if not form.validate_on_submit():
        if request_wants_json():
            return flask.jsonify({'errors': form.errors}), 400
        else:
            return flask.render_template('datasets/images/classification/new.html', form=form), 400

    job = None
    try:
        job = ImageClassificationDatasetJob(
            username=utils.auth.get_username(),
            name=form.dataset_name.data,
            group=form.group_name.data,
            image_dims=(
                int(form.resize_height.data),
                int(form.resize_width.data),
                int(form.resize_channels.data),
            ),
            resize_mode=form.resize_mode.data
        )

        if form.method.data == 'folder':
            from_folders(job, form)

        elif form.method.data == 'textfile':
            from_files(job, form)

        else:
            raise ValueError('method not supported')

        # Save form data with the job so we can easily clone it later.
        save_form_to_job(job, form)

        scheduler.add_job(job)
        if request_wants_json():
            return flask.jsonify(job.json_dict())
        else:
            return flask.redirect(flask.url_for('digits.dataset.views.show', job_id=job.id()))

    except:
        if job:
            scheduler.delete_job(job)
        raise


def show(job, related_jobs=None):
    """
    Called from digits.dataset.views.datasets_show()
    """
    return flask.render_template('datasets/images/classification/show.html', job=job, related_jobs=related_jobs)


def summary(job):
    """
    Return a short HTML summary of an ImageClassificationDatasetJob
    """
    return flask.render_template('datasets/images/classification/summary.html',
                                 dataset=job)


@blueprint.route('/explore', methods=['GET'])
def explore():
    """
    Returns a gallery consisting of the images of one of the dbs
    """
    job = job_from_request()
    # Get LMDB
    db = flask.request.args.get('db', 'train')
    if 'train' in db.lower():
        task = job.train_db_task()
    elif 'val' in db.lower():
        task = job.val_db_task()
    elif 'test' in db.lower():
        task = job.test_db_task()
    if task is None:
        raise ValueError('No create_db task for {0}'.format(db))
    if task.status != 'D':
        raise ValueError("This create_db task's status should be 'D' but is '{0}'".format(task.status))
    if task.backend != 'lmdb':
        raise ValueError("Backend is {0} while expected backend is lmdb".format(task.backend))
    db_path = job.path(task.db_name)
    labels = task.get_labels()

    page = int(flask.request.args.get('page', 0))
    size = int(flask.request.args.get('size', 25))
    label = flask.request.args.get('label', None)

    if label is not None:
        try:
            label = int(label)
        except ValueError:
            label = None

    reader = DbReader(db_path)
    count = 0
    imgs = []

    min_page = max(0, page - 5)
    if label is None:
        total_entries = reader.total_entries
    else:
        total_entries = task.distribution[str(label)]

    max_page = min((total_entries - 1) / size, page + 5)
    pages = range(min_page, max_page + 1)
    for key, value in reader.entries():
        if count >= page * size:
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value)
            if label is None or datum.label == label:
                if datum.encoded:
                    s = StringIO()
                    s.write(datum.data)
                    s.seek(0)
                    img = PIL.Image.open(s)
                else:
                    import caffe.io
                    arr = caffe.io.datum_to_array(datum)
                    # CHW -> HWC
                    arr = arr.transpose((1, 2, 0))
                    if arr.shape[2] == 1:
                        # HWC -> HW
                        arr = arr[:, :, 0]
                    elif arr.shape[2] == 3:
                        # BGR -> RGB
                        # XXX see issue #59
                        arr = arr[:, :, [2, 1, 0]]
                    img = PIL.Image.fromarray(arr)
                imgs.append({"label": labels[datum.label], "b64": utils.image.embed_image_html(img)})
        if label is None:
            count += 1
        else:
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value)
            if datum.label == int(label):
                count += 1
        if len(imgs) >= size:
            break

    return flask.render_template(
        'datasets/images/explore.html',
        page=page, size=size, job=job, imgs=imgs, labels=labels,
        pages=pages, label=label, total_entries=total_entries, db=db)
