# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import flask
import werkzeug.exceptions

from . import images as dataset_images
from digits.utils.routing import request_wants_json
from digits.webapp import app, scheduler

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

    if request_wants_json():
        return flask.jsonify(job.json_dict(True))
    else:
        if isinstance(job, dataset_images.ImageClassificationDatasetJob):
            return dataset_images.classification.views.show(job)
        elif isinstance(job, dataset_images.GenericImageDatasetJob):
            return dataset_images.generic.views.show(job)
        else:
            raise werkzeug.exceptions.BadRequest('Invalid job type')

