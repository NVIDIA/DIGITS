# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import json
import traceback

import flask
from werkzeug import HTTP_STATUS_CODES
import werkzeug.exceptions
from flask.ext.socketio import join_room, leave_room

from . import dataset, model
from config import config_value
from webapp import app, socketio, scheduler, autodoc
import dataset.views
import model.views
from digits.utils import errors
from digits.utils.routing import request_wants_json

@app.route('/index.json', methods=['GET'])
@app.route('/', methods=['GET'])
@autodoc(['home', 'api'])
def home():
    """
    DIGITS home page
    Returns information about each job on the server

    Returns JSON when requested:
        {
            datasets: [{id, name, status},...],
            models: [{id, name, status},...]
        }
    """
    running_datasets    = get_job_list(dataset.DatasetJob, True)
    completed_datasets  = get_job_list(dataset.DatasetJob, False)
    running_models      = get_job_list(model.ModelJob, True)
    completed_models    = get_job_list(model.ModelJob, False)

    if request_wants_json():
        return flask.jsonify({
            'datasets': [j.json_dict()
                for j in running_datasets + completed_datasets],
            'models': [j.json_dict()
                for j in running_models + completed_models],
            })
    else:
        new_dataset_options = [
                ('Images', [
                    {
                        'title': 'Classification',
                        'id': 'image-classification',
                        'url': flask.url_for('image_classification_dataset_new'),
                        },
                    ])
                ]
        new_model_options = [
                ('Images', [
                    {
                        'title': 'Classification',
                        'id': 'image-classification',
                        'url': flask.url_for('image_classification_model_new'),
                        },
                    ])
                ]

        return flask.render_template('home.html',
                new_dataset_options = new_dataset_options,
                running_datasets    = running_datasets,
                completed_datasets  = completed_datasets,
                new_model_options   = new_model_options,
                running_models      = running_models,
                completed_models    = completed_models,
                )

def get_job_list(cls, running):
    return sorted(
            [j for j in scheduler.jobs if isinstance(j, cls) and j.status.is_running() == running],
            key=lambda j: j.status_history[0][1],
            reverse=True,
            )


### Jobs routes

@app.route('/jobs/<job_id>', methods=['GET'])
@autodoc('jobs')
def show_job(job_id):
    """
    Redirects to the appropriate /datasets/ or /models/ page
    """
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    if isinstance(job, dataset.DatasetJob):
        return flask.redirect(flask.url_for('datasets_show', job_id=job_id))
    if isinstance(job, model.ModelJob):
        return flask.redirect(flask.url_for('models_show', job_id=job_id))
    else:
        raise werkzeug.exceptions.BadRequest('Invalid job type')

@app.route('/jobs/<job_id>', methods=['PUT'])
@autodoc('jobs')
def edit_job(job_id):
    """
    Edit the name of a job
    """
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    old_name = job.name()
    job._name = flask.request.form['job_name']
    return 'Changed job name from "%s" to "%s"' % (old_name, job.name())

@app.route('/datasets/<job_id>/status', methods=['GET'])
@app.route('/models/<job_id>/status', methods=['GET'])
@app.route('/jobs/<job_id>/status', methods=['GET'])
@autodoc('jobs')
def job_status(job_id):
    """
    Returns a JSON objecting representing the status of a job
    """
    job = scheduler.get_job(job_id)
    result = {}
    if job is None:
        result['error'] = 'Job not found.'
    else:
        result['error'] = None
        result['status'] = job.status.name
        result['name'] = job.name()
        result['type'] = job.job_type()
    return json.dumps(result)

@app.route('/datasets/<job_id>', methods=['DELETE'])
@app.route('/models/<job_id>', methods=['DELETE'])
@app.route('/jobs/<job_id>', methods=['DELETE'])
@autodoc('jobs')
def delete_job(job_id):
    """
    Deletes a job
    """
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    try:
        if scheduler.delete_job(job_id):
            return 'Job deleted.'
        else:
            raise werkzeug.exceptions.Forbidden('Job not deleted')
    except errors.DeleteError as e:
        raise werkzeug.exceptions.Forbidden(str(e))

@app.route('/datasets/<job_id>/abort', methods=['POST'])
@app.route('/models/<job_id>/abort', methods=['POST'])
@app.route('/jobs/<job_id>/abort', methods=['POST'])
@autodoc('jobs')
def abort_job(job_id):
    """
    Aborts a running job
    """
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    if scheduler.abort_job(job_id):
        return 'Job aborted.'
    else:
        raise werkzeug.exceptions.Forbidden('Job not aborted')

### Error handling

@app.errorhandler(Exception)
def handle_error(e):
    """
    Handle errors, formatting them as JSON if requested
    """
    error_type = type(e).__name__
    message = str(e)
    trace = None
    description = None
    status_code = 500
    if isinstance(e, werkzeug.exceptions.HTTPException):
        status_code = e.code
        description = e.description
    if app.debug:
        trace = traceback.format_exc()

    if request_wants_json():
        details = {
                'message': message,
                'type': error_type,
                }
        if description is not None:
            details['description'] = description
        if trace is not None:
            details['trace'] = trace.split('\n')
        return flask.jsonify({'error': details}), status_code
    else:
        return flask.render_template('error.html',
                title       = error_type,
                message     = message,
                description = description,
                trace       = trace,
                ), status_code

# Register this handler for all error codes
# Necessary for flask<=0.10.1
for code in HTTP_STATUS_CODES:
    app.register_error_handler(code, handle_error)

### File serving

@app.route('/files/<path:path>', methods=['GET'])
@autodoc('util')
def serve_file(path):
    """
    Return a file in the jobs directory

    If you install the nginx.site file, nginx will serve files instead
    and this path will never be used
    """
    jobs_dir = config_value('jobs_dir')
    path = os.path.normpath(os.path.join(jobs_dir, path))

    # Don't allow path manipulation
    if not os.path.commonprefix([path, jobs_dir]).startswith(jobs_dir):
        raise werkzeug.exceptions.Forbidden('Path manipulation not allowed')

    if not os.path.exists(path):
        raise werkzeug.exceptions.NotFound('File not found')
    if os.path.isdir(path):
        raise werkzeug.exceptions.Forbidden('Folder cannot be served')

    with open(path, 'r') as infile:
        response = flask.make_response(infile.read())
        response.headers["Content-Disposition"] = "attachment; filename=%s" % os.path.basename(path)
        return response

### SocketIO functions

## /home

@socketio.on('connect', namespace='/home')
def on_connect():
    """
    Somebody connected to the homepage
    """
    pass

@socketio.on('disconnect', namespace='/home')
def on_disconnect():
    """
    Somebody disconnected from the homepage
    """
    pass

## /jobs

@socketio.on('connect', namespace='/jobs')
def on_connect():
    """
    Somebody connected to a jobs page
    """

@socketio.on('disconnect', namespace='/jobs')
def on_disconnect():
    """
    Somebody disconnected from a jobs page
    """
    pass

@socketio.on('join', namespace='/jobs')
def on_join(data):
    """
    Somebody joined a room
    """
    room = data['room']
    join_room(room)
    flask.session['room'] = room

@socketio.on('leave', namespace='/jobs')
def on_leave():
    """
    Somebody left a room
    """
    if 'room' in flask.session:
        room = flask.session['room']
        del flask.session['room']
        #print '>>> Somebody left room %s' % room
        leave_room(room)

