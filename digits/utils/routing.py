# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import flask
import werkzeug.exceptions


def job_from_request():
    """
    Returns the job after grabbing job_id from request.args or request.form
    Raises werkzeug.exceptions
    """
    from digits.webapp import scheduler

    job_id = get_request_arg('job_id')
    if job_id is None:
        raise werkzeug.exceptions.BadRequest('job_id is a required field')

    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')
    else:
        return job

# Adapted from http://flask.pocoo.org/snippets/45/


def request_wants_json():
    """
    Returns True if the response should be JSON
    """
    if flask.request.base_url.endswith('.json'):
        return True
    best = flask.request.accept_mimetypes \
        .best_match(['application/json', 'text/html'])
    # Some browsers accept on */* and we don't want to deliver JSON to an ordinary browser
    return best == 'application/json' and \
        flask.request.accept_mimetypes[best] > \
        flask.request.accept_mimetypes['text/html']

# check for arguments pass to url


def get_request_arg(key):
    value = None
    if key in flask.request.args:
        value = flask.request.args[key]
    elif key in flask.request.form:
        value = flask.request.form[key]
    return value
