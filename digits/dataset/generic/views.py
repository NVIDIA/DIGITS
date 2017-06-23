# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

import caffe_pb2
import flask
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

from .forms import GenericDatasetForm
from .job import GenericDatasetJob
from digits import extensions, utils
from digits.utils.constants import COLOR_PALETTE_ATTRIBUTE
from digits.utils.routing import request_wants_json, job_from_request
from digits.utils.lmdbreader import DbReader
from digits.webapp import scheduler

blueprint = flask.Blueprint(__name__, __name__)


@blueprint.route('/new/<extension_id>', methods=['GET'])
@utils.auth.requires_login
def new(extension_id):
    """
    Returns a form for a new GenericDatasetJob
    """

    form = GenericDatasetForm()

    # Is there a request to clone a job with ?clone=<job_id>
    utils.forms.fill_form_if_cloned(form)

    extension = extensions.data.get_extension(extension_id)
    if extension is None:
        raise ValueError("Unknown extension '%s'" % extension_id)
    extension_form = extension.get_dataset_form()

    # Is there a request to clone a job with ?clone=<job_id>
    utils.forms.fill_form_if_cloned(extension_form)

    template, context = extension.get_dataset_template(extension_form)
    rendered_extension = flask.render_template_string(template, **context)

    return flask.render_template(
        'datasets/generic/new.html',
        extension_title=extension.get_title(),
        extension_id=extension_id,
        extension_html=rendered_extension,
        form=form
    )


@blueprint.route('/create/<extension_id>.json', methods=['POST'])
@blueprint.route('/create/<extension_id>',
                 methods=['POST'],
                 strict_slashes=False)
@utils.auth.requires_login(redirect=False)
def create(extension_id):
    """
    Creates a new GenericDatasetJob

    Returns JSON when requested: {job_id,name,status} or {errors:[]}
    """
    form = GenericDatasetForm()
    form_valid = form.validate_on_submit()

    extension_class = extensions.data.get_extension(extension_id)
    extension_form = extension_class.get_dataset_form()
    extension_form_valid = extension_form.validate_on_submit()

    if not (extension_form_valid and form_valid):
        # merge errors
        errors = form.errors.copy()
        errors.update(extension_form.errors)

        template, context = extension_class.get_dataset_template(
            extension_form)
        rendered_extension = flask.render_template_string(
            template,
            **context)

        if request_wants_json():
            return flask.jsonify({'errors': errors}), 400
        else:
            return flask.render_template(
                'datasets/generic/new.html',
                extension_title=extension_class.get_title(),
                extension_id=extension_id,
                extension_html=rendered_extension,
                form=form,
                errors=errors), 400

    # create instance of extension class
    extension = extension_class(**extension_form.data)

    job = None
    try:
        # create job
        job = GenericDatasetJob(
            username=utils.auth.get_username(),
            name=form.dataset_name.data,
            group=form.group_name.data,
            backend=form.dsopts_backend.data,
            feature_encoding=form.dsopts_feature_encoding.data,
            label_encoding=form.dsopts_label_encoding.data,
            batch_size=int(form.dsopts_batch_size.data),
            num_threads=int(form.dsopts_num_threads.data),
            force_same_shape=form.dsopts_force_same_shape.data,
            extension_id=extension_id,
            extension_userdata=extension.get_user_data(),
        )

        # Save form data with the job so we can easily clone it later.
        utils.forms.save_form_to_job(job, form)
        utils.forms.save_form_to_job(job, extension_form)

        # schedule tasks
        scheduler.add_job(job)

        if request_wants_json():
            return flask.jsonify(job.json_dict())
        else:
            return flask.redirect(flask.url_for(
                'digits.dataset.views.show',
                job_id=job.id()))

    except:
        if job:
            scheduler.delete_job(job)
        raise


@blueprint.route('/explore', methods=['GET'])
def explore():
    """
    Returns a gallery consisting of the images of one of the dbs
    """
    job = job_from_request()
    # Get LMDB
    db = job.path(flask.request.args.get('db'))
    db_path = job.path(db)

    if (os.path.basename(db_path) == 'labels' and
            COLOR_PALETTE_ATTRIBUTE in job.extension_userdata and
            job.extension_userdata[COLOR_PALETTE_ATTRIBUTE]):
        # assume single-channel 8-bit palette
        palette = job.extension_userdata[COLOR_PALETTE_ATTRIBUTE]
        palette = np.array(palette).reshape((len(palette) / 3, 3)) / 255.
        # normalize input pixels to [0,1]
        norm = mpl.colors.Normalize(vmin=0, vmax=255)
        # create map
        cmap = plt.cm.ScalarMappable(norm=norm,
                                     cmap=mpl.colors.ListedColormap(palette))
    else:
        cmap = None

    page = int(flask.request.args.get('page', 0))
    size = int(flask.request.args.get('size', 25))

    reader = DbReader(db_path)
    count = 0
    imgs = []

    min_page = max(0, page - 5)
    total_entries = reader.total_entries

    max_page = min((total_entries - 1) / size, page + 5)
    pages = range(min_page, max_page + 1)
    for key, value in reader.entries():
        if count >= page * size:
            datum = caffe_pb2.Datum()
            datum.ParseFromString(value)
            if not datum.encoded:
                raise RuntimeError("Expected encoded database")
            s = StringIO()
            s.write(datum.data)
            s.seek(0)
            img = PIL.Image.open(s)
            if cmap and img.mode in ['L', '1']:
                data = np.array(img)
                data = cmap.to_rgba(data) * 255
                data = data.astype('uint8')
                # keep RGB values only, remove alpha channel
                data = data[:, :, 0:3]
                img = PIL.Image.fromarray(data)
            imgs.append({"label": None, "b64": utils.image.embed_image_html(img)})
        count += 1
        if len(imgs) >= size:
            break

    return flask.render_template(
        'datasets/images/explore.html',
        page=page, size=size, job=job, imgs=imgs, labels=None,
        pages=pages, label=None, total_entries=total_entries, db=db)


def show(job, related_jobs=None):
    """
    Called from digits.dataset.views.show()
    """
    return flask.render_template('datasets/generic/show.html', job=job, related_jobs=related_jobs)


def summary(job):
    """
    Return a short HTML summary of a GenericDatasetJob
    """
    return flask.render_template('datasets/generic/summary.html', dataset=job)
