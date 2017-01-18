# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import flask
import werkzeug.exceptions

from . import images as dataset_images
from . import generic
from digits.utils.routing import job_from_request, request_wants_json
from digits.webapp import scheduler

blueprint = flask.Blueprint(__name__, __name__)


@blueprint.route('/<job_id>.json', methods=['GET'])
@blueprint.route('/<job_id>', methods=['GET'])
def show(job_id):
    """
    Show a DatasetJob

    Returns JSON when requested:
        {id, name, directory, status}
    """
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    related_jobs = scheduler.get_related_jobs(job)

    if request_wants_json():
        return flask.jsonify(job.json_dict(True))
    else:
        if isinstance(job, dataset_images.ImageClassificationDatasetJob):
            return dataset_images.classification.views.show(job, related_jobs=related_jobs)
        elif isinstance(job, dataset_images.GenericImageDatasetJob):
            return dataset_images.generic.views.show(job, related_jobs=related_jobs)
        elif isinstance(job, generic.GenericDatasetJob):
            return generic.views.show(job, related_jobs=related_jobs)
        else:
            raise werkzeug.exceptions.BadRequest('Invalid job type')


@blueprint.route('/summary', methods=['GET'])
def summary():
    """
    Return a short HTML summary of a DatasetJob
    """
    job = job_from_request()
    if isinstance(job, dataset_images.ImageClassificationDatasetJob):
        return dataset_images.classification.views.summary(job)
    elif isinstance(job, dataset_images.GenericImageDatasetJob):
        return dataset_images.generic.views.summary(job)
    elif isinstance(job, generic.GenericDatasetJob):
        return generic.views.summary(job)
    else:
        raise werkzeug.exceptions.BadRequest('Invalid job type')
