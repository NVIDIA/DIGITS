# -*- coding: utf-8 -*-

import flask
import werkzeug.exceptions
from digits.webapp import app, scheduler, autodoc
from digits.utils.routing import request_wants_json
from flask import render_template, request, url_for, flash, make_response, abort, jsonify

import digits
import images as evaluation_images
import images.views

NAMESPACE = '/evaluations/'

@app.route(NAMESPACE + '<job_id>.json', methods=['GET'])
@app.route(NAMESPACE + '<job_id>', methods=['GET'])
@autodoc('evaluations')
def evaluations_show(job_id):
    """
    Show an EvaluationJob

    Returns JSON when requested:
        {id, name, directory, status }
    """
    job = scheduler.get_job(job_id)

    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')
    related_jobs = scheduler.get_related_jobs(job)

    if request_wants_json():
        return flask.jsonify(job.json_dict(True))

    if isinstance(job, evaluation_images.ImageEvaluationJob):
        return evaluation_images.classification.views.show(job, related_jobs=related_jobs)
    else:
        raise werkzeug.exceptions.BadRequest(
                'Invalid job type')
