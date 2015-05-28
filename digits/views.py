# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import json
import traceback

from flask import render_template, flash, redirect, session, url_for, abort, make_response, request, jsonify
from flask.ext.socketio import emit, join_room, leave_room

from . import dataset, model
from config import config_value
from webapp import app, socketio, scheduler, autodoc
from status import Status
import dataset.views
import model.views
from digits.utils import errors

@app.route('/')
@autodoc('home')
def home():
    """
    DIGITS home page
    Displays all datasets and models on the server and their status
    """
    new_dataset_options = [
            ('Images', [
                {
                    'title': 'Classification',
                    'id': 'image-classification',
                    'url': url_for('image_classification_dataset_new'),
                    },
                ])
            ]
    new_model_options = [
            ('Images', [
                {
                    'title': 'Classification',
                    'id': 'image-classification',
                    'url': url_for('image_classification_model_new'),
                    },
                ])
            ]
    return render_template('home.html',
            new_dataset_options = new_dataset_options,
            running_datasets    = get_job_list(dataset.DatasetJob, True),
            completed_datasets  = get_job_list(dataset.DatasetJob, False),
            new_model_options   = new_model_options,
            running_models      = get_job_list(model.ModelJob, True),
            completed_models    = get_job_list(model.ModelJob, False),
            )

@app.route('/index.json')
@autodoc('home')
def home_json():
    """
    JSON version of the DIGITS home page
    Returns information about each job on the server
    """
    datasets = get_job_list(dataset.DatasetJob, True) + get_job_list(dataset.DatasetJob, False)
    datasets = [{
        'name': j.name(),
        'id': j.id(),
        'status': j.status.name,
        } for j in datasets]
    models = get_job_list(model.ModelJob, True) + get_job_list(model.ModelJob, False)
    models = [{
        'name': j.name(),
        'id': j.id(),
        'status': j.status.name,
        } for j in models]
    return jsonify({
        'datasets': datasets,
        'models': models,
        })

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
        abort(404)

    if isinstance(job, dataset.DatasetJob):
        return redirect(url_for('datasets_show', job_id=job_id))
    if isinstance(job, model.ModelJob):
        return redirect(url_for('models_show', job_id=job_id))
    else:
        abort(404)

@app.route('/jobs/<job_id>', methods=['PUT'])
@autodoc('jobs')
def edit_job(job_id):
    """
    Edit the name of a job
    """
    job = scheduler.get_job(job_id)

    if job is None:
        abort(404)

    old_name = job.name()
    job._name = request.form['job_name']
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
    if not job:
        return 'Job not found!', 404
    try:
        if scheduler.delete_job(job_id):
            return 'Job deleted.'
        else:
            return 'Job could not be deleted! Check log for more details', 403
    except errors.DeleteError as e:
        return e.__str__(), 403

@app.route('/datasets/<job_id>/abort', methods=['POST'])
@app.route('/models/<job_id>/abort', methods=['POST'])
@app.route('/jobs/<job_id>/abort', methods=['POST'])
@autodoc('jobs')
def abort_job(job_id):
    """
    Aborts a running job
    """
    if scheduler.abort_job(job_id):
        return 'Job aborted.'
    else:
        return 'Job not found!', 404

### Error handling

@app.errorhandler(Exception)
def handle_exception(e, status_code=500):
    if 'DIGITS_MODE_TEST' in os.environ:
        raise e
    title = type(e).__name__
    message = str(e)
    trace = None
    if app.debug:
        trace = traceback.format_exc()
        #trace = '<br>\n'.join(trace.split('\n'))
    return render_template('500.html',
            title   = title,
            message = message,
            trace   = trace,
            ), status_code

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
        abort(403)

    if not os.path.exists(path):
        abort(404)
    if os.path.isdir(path):
        abort(403)

    with open(path, 'r') as infile:
        response = make_response(infile.read())
        response.headers["Content-Disposition"] = "attachment; filename=%s" % os.path.basename(path)
        return response

### SocketIO functions

## /home

@socketio.on('connect', namespace='/home')
def on_connect():
    #print '>>> Somebody connected to the homepage'
    pass

@socketio.on('disconnect', namespace='/home')
def on_disconnect():
    #print '>>> Somebody disconnected from the homepage'
    pass

## /jobs

@socketio.on('connect', namespace='/jobs')
def on_connect():
    #print '>>> Somebody connected to a jobs page'
    pass

@socketio.on('disconnect', namespace='/jobs')
def on_disconnect():
    #print '>>> Somebody disconnected from a jobs page'
    pass

@socketio.on('join', namespace='/jobs')
def on_join(data):
    room = data['room']
    #print '>>> Somebody joined room %s' % room
    join_room(room)
    session['room'] = room

@socketio.on('leave', namespace='/jobs')
def on_leave():
    if 'room' in session:
        room = session['room']
        del session['room']
        #print '>>> Somebody left room %s' % room
        leave_room(room)

