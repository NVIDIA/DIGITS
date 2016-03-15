# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import flask
import numpy as np
import werkzeug.exceptions

from . import images as inference_images
from digits.webapp import scheduler

blueprint = flask.Blueprint(__name__, __name__)

@blueprint.route('/<job_id>.json', methods=['GET'])
@blueprint.route('/<job_id>', methods=['GET'])
def show(job_id):
    """
    Show a ModelJob

    Returns JSON when requested:
        {id, name, directory, status, snapshots: [epoch,epoch,...]}
    """
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    # show appropriate visualization
    if isinstance(job, inference_images.ImageInferenceClassificationJob):
        return inference_images.classification.views.show(job)
    else:
        raise werkzeug.exceptions.BadRequest(
                'Invalid job type')
