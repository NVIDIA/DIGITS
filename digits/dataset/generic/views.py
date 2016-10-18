# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

import json
import flask

from .forms import GenericDatasetForm
from .job import GenericDatasetJob
from digits import extensions, utils
from digits.utils.routing import request_wants_json, job_from_request, get_request_arg
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


def get_database_visualizations(dataset, inputs, outputs):
    # form data may be passed through the query
    form_data = get_request_arg('form_data')
    if form_data:
        form_data = json.loads(form_data)

    # get extension ID from form and retrieve extension class
    from_data = form_data and 'view_extension_id' in form_data
    if from_data:
        view_extension_id = form_data['view_extension_id']
    else:
        view_extension_id = get_request_arg('view_extension_id')

    if view_extension_id:
        extension_class = extensions.view.get_extension(view_extension_id)
        if extension_class is None:
            raise ValueError("Unknown extension '%s'" % view_extension_id)
    else:
        # no view extension specified, use default
        extension_class = extensions.view.get_default_extension()

    if from_data:
        data = form_data
    else:
        extension_form = extension_class.get_config_form()
        # validate form
        extension_form_valid = extension_form.validate_on_submit()
        if not extension_form_valid:
            raise ValueError("Extension form validation failed with %s" % repr(extension_form.errors))
        data = extension_form.data

    # create instance of extension class
    extension = extension_class(dataset, **data)

    visualizations = []
    # process data
    n = len(inputs['ids'])
    for idx in range(n):
        input_id = inputs['ids'][idx]
        input_data = inputs['data'][idx]
        output_data = {key: outputs[key][idx] for key in outputs}
        data = extension.process_data(
            input_id,
            input_data,
            output_data)
        template, context = extension.get_view_template(data)
        visualizations.append(
            flask.render_template_string(template, **context))
    # get header
    template, context = extension.get_header_template()
    header = flask.render_template_string(template, **context) if template else None
    app_begin, app_end = extension.get_ng_templates()
    return visualizations, header, app_begin, app_end


@blueprint.route('/explore', methods=['GET', 'POST'])
def explore():
    """
    Returns a gallery consisting of the images from the view extension
    """
    job = job_from_request()
    page = int(flask.request.args.get('page', 0))
    size = int(flask.request.args.get('size', 25))

    extension = extensions.data.get_extension(job.extension_id)
    if extension is None:
        raise ValueError("Unknown extension '%s'" % job.extension_id)

    # Get LMDB(s)
    feature_db = job.path(flask.request.args.get('feature_db'))
    feature_db_path = job.path(feature_db) if feature_db else None

    label_db = job.path(flask.request.args.get('label_db'))
    label_db_path = job.path(label_db) if label_db else None

    # Get data from data extension
    inputs, outputs, total_entries = extension.get_data(feature_db_path, label_db_path, page, size)
    # Get visualization from the view extension
    visualizations, header, app_begin, app_end = get_database_visualizations(job, inputs, outputs)

    min_page = max(0, page - 5)
    max_page = min((total_entries - 1) / size, page + 5)
    pages = range(min_page, max_page + 1)

    # This is weak, but should do the job. This allows the form data to
    # be passed to subsequent pages through the query
    form_data = get_request_arg('form_data')
    if not form_data:
        form_data = dict(zip(flask.request.form.keys(), flask.request.form.values()))
        form_data = json.dumps(form_data)

    return flask.render_template('datasets/images/explore.html',
                                 page=page, size=size, job=job, labels=None,
                                 pages=pages, label=None, total_entries=total_entries,
                                 feature_db=feature_db, label_db=label_db,
                                 form_data=form_data,
                                 header=header,
                                 app_begin=app_begin,
                                 app_end=app_end,
                                 visualizations=visualizations,
                                 )


def get_view_extensions():
    """
    return all enabled view extensions
    """
    view_extensions = {}
    all_extensions = extensions.view.get_extensions()
    for extension in all_extensions:
        view_extensions[extension.get_id()] = extension.get_title()
    return view_extensions


def show(job, related_jobs=None):
    """
    Called from digits.dataset.views.show()
    """
    data_extension = extensions.data.get_extension(job.extension_id)
    can_explore = (data_extension and data_extension.can_explore())
    view_extensions = get_view_extensions()
    return flask.render_template('datasets/generic/show.html',
                                 job=job,
                                 related_jobs=related_jobs,
                                 view_extensions=view_extensions,
                                 can_explore=can_explore,
                                 )


def summary(job):
    """
    Return a short HTML summary of a GenericDatasetJob
    """
    return flask.render_template('datasets/generic/summary.html', dataset=job)
