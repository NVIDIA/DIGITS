# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.

import flask
import werkzeug.exceptions

from digits.webapp import app, scheduler, autodoc
from digits.utils.routing import request_wants_json, get_workspace
import images.views
import images as dataset_images

NAMESPACE = '/datasets/'

@app.route(NAMESPACE + '<job_id>.json', methods=['GET'])
@app.route(NAMESPACE + '<job_id>', methods=['GET'])
@autodoc(['datasets', 'api'])
def datasets_show(job_id):
    """
    Show a DatasetJob

    Returns JSON when requested:
        {id, name, directory, status}
    """
    workspace = get_workspace(flask.request.url)
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    if request_wants_json():
        return flask.jsonify(job.json_dict(True))
    else:
        if isinstance(job, dataset_images.ImageClassificationDatasetJob):
            return dataset_images.classification.views.show(job, workspace)
        elif isinstance(job, dataset_images.GenericImageDatasetJob):
            return dataset_images.generic.views.show(job, workspace)
        else:
            raise werkzeug.exceptions.BadRequest('Invalid job type')

