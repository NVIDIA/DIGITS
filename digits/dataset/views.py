# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import flask

from digits.webapp import app, scheduler, autodoc
from digits.utils.routing import request_wants_json
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
    job = scheduler.get_job(job_id)

    if job is None:
        flask.abort(404)

    if request_wants_json():
        return flask.jsonify(job.json_dict(True))
    else:
        if isinstance(job, dataset_images.ImageClassificationDatasetJob):
            return dataset_images.classification.views.show(job)
        else:
            flask.abort(404)

@app.route(NAMESPACE + 'summary', methods=['GET'])
@autodoc('datasets')
def dataset_summary():
    """
    Return a short HTML summary of a DatasetJob
    """
    job_id = flask.request.args.get('job_id', '')
    if not job_id:
        return 'No job_id in request!'

    job = scheduler.get_job(job_id)

    return flask.render_template('datasets/summary.html', dataset=job)

