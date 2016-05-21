# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import flask

from .forms import GenericDatasetForm
from .job import GenericDatasetJob

from digits import extensions, utils
from digits.utils.routing import request_wants_json
from digits.webapp import scheduler

blueprint = flask.Blueprint(__name__, __name__)


@blueprint.route('/new/<extension_id>', methods=['GET'])
@utils.auth.requires_login
def new(extension_id):
    """
    Returns a form for a new GenericDatasetJob
    """

    form = GenericDatasetForm()

    ## Is there a request to clone a job with ?clone=<job_id>
    utils.forms.fill_form_if_cloned(form)

    extension = extensions.data.get_extension(extension_id)
    if extension is None:
        raise ValueError("Unknown extension '%s'" % extension_id)
    extension_form = extension.get_dataset_form()

    ## Is there a request to clone a job with ?clone=<job_id>
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
            backend=form.dsopts_backend.data,
            feature_encoding=form.dsopts_feature_encoding.data,
            label_encoding=form.dsopts_label_encoding.data,
            batch_size=int(form.dsopts_batch_size.data),
            num_threads=int(form.dsopts_num_threads.data),
            extension_id=extension_id,
            extension_userdata=extension.get_user_data(),
            )

        ## Save form data with the job so we can easily clone it later.
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
