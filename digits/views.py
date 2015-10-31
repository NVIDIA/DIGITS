# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import json
import traceback
import glob
import platform

import flask
from werkzeug import HTTP_STATUS_CODES
import werkzeug.exceptions
from flask.ext.socketio import join_room, leave_room

import digits
from . import dataset, model
from config import config_value
from webapp import app, socketio, scheduler, autodoc
import dataset.views
import model.views
from digits.utils import errors
from digits.utils.routing import request_wants_json
from digits.log import logger
from flask import Flask, request, render_template
from workspaces import get_workspace
import shutil
import os

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
    jobs_dir = config_value('jobs_dir')
    default_workspace = jobs_dir+"/"+"Default"
    if not os.path.exists(default_workspace):
        os.makedirs(default_workspace)
    workspace = get_workspace(flask.request.url)
    running_datasets    = get_job_list(dataset.DatasetJob, True, workspace)
    completed_datasets  = get_job_list(dataset.DatasetJob, False, workspace)
    running_models      = get_job_list(model.ModelJob, True, workspace)
    completed_models    = get_job_list(model.ModelJob, False, workspace)

    if request_wants_json():
        data = {
                'version': digits.__version__,
                'jobs_dir': config_value('jobs_dir'),
                'datasets': [j.json_dict()
                    for j in running_datasets + completed_datasets],
                'models': [j.json_dict()
                    for j in running_models + completed_models],
                }
        if config_value('server_name'):
            data['server_name'] = config_value('server_name')
        return flask.jsonify(data)
    else:
        new_dataset_options = [
                ('Images', [
                    {
                        'title': 'Classification',
                        'id': 'image-classification',
                        'url': flask.url_for('image_classification_dataset_new'),
                        },
                    {
                        'title': 'Other',
                        'id': 'image-generic',
                        'url': flask.url_for('generic_image_dataset_new'),
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
                    {
                        'title': 'Other',
                        'id': 'image-generic',
                        'url': flask.url_for('generic_image_model_new'),
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
                total_gpu_count     = len(scheduler.resources['gpus']),
                remaining_gpu_count = sum(r.remaining() for r in scheduler.resources['gpus']),
                workspace = workspace,
                )

def get_job_list(cls, running, *args):
    scheduler_jobs = [j._id for j in scheduler.jobs if isinstance(j, cls) and j.status.is_running() == running]
    workspace_dir = os.path.join(config_value('jobs_dir'), args[0])
    workspace_jobs = next(os.walk(workspace_dir))[1]
    workspace_scheduled_jobs = list(set(workspace_jobs) & set(scheduler_jobs))
    return sorted(
            [j for j in scheduler.jobs if isinstance(j, cls) and j.status.is_running() == running and j._id in workspace_scheduled_jobs],
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
    workspace = get_workspace(flask.request.url)
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    if isinstance(job, dataset.DatasetJob):
        return flask.redirect(flask.url_for('datasets_show', job_id=job_id)+'?workspace='+workspace)
    if isinstance(job, model.ModelJob):
        return flask.redirect(flask.url_for('models_show', job_id=job_id)+'?workspace='+workspace)
    else:
        raise werkzeug.exceptions.BadRequest('Invalid job type')

@app.route('/jobs/<job_id>', methods=['PUT'])
@autodoc('jobs')
def edit_job(job_id):
    """
    Edit a job's name and/or notes
    """
    workspace = get_workspace(flask.request.url)
    job = scheduler.get_job(job_id)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    # Edit name
    if 'job_name' in flask.request.form:
        name = flask.request.form['job_name'].strip()
        if not name:
            raise werkzeug.exceptions.BadRequest('name cannot be blank')
        job._name = name
        logger.info('Set name to "%s".' % job.name(), job_id=job.id())

    # Edit notes
    if 'job_notes' in flask.request.form:
        notes = flask.request.form['job_notes'].strip()
        if not notes:
            notes = None
        job._notes = notes
        logger.info('Updated notes.', job_id=job.id())

    return '%s updated.' % job.job_type()

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
    workspace = get_workspace(flask.request.url)
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

@app.route('/clone/<clone>', methods=['POST', 'GET'])
@autodoc('jobs')
def clone_job(clone):
    """
    Clones a job with the id <clone>, populating the creation page with data saved in <clone>
    """

    ## <clone> is the job_id to clone

    job = scheduler.get_job(clone)
    if job is None:
        raise werkzeug.exceptions.NotFound('Job not found')

    if isinstance(job, dataset.ImageClassificationDatasetJob):
        return flask.redirect(flask.url_for('image_classification_dataset_new') + '?clone=' + clone)
    if isinstance(job, dataset.GenericImageDatasetJob):
        return flask.redirect(flask.url_for('generic_image_dataset_new') + '?clone=' + clone)
    if isinstance(job, model.ImageClassificationModelJob):
        return flask.redirect(flask.url_for('image_classification_model_new') + '?clone=' + clone)
    if isinstance(job, model.GenericImageModelJob):
        return flask.redirect(flask.url_for('generic_image_model_new') + '?clone=' + clone)
    else:
        raise werkzeug.exceptions.BadRequest('Invalid job type')

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
    if code not in [301]:
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
    return flask.send_from_directory(jobs_dir, path)

### Path Completion

@app.route('/autocomplete/path', methods=['GET'])
@autodoc('util')
def path_autocomplete():
    """
    Return a list of paths matching the specified preamble

    """
    path = flask.request.args.get('query','')
    suggestions = glob.glob(path+"*")
    if platform.system() == 'Windows':
        # on windows, convert backslashes with forward slashes
        suggestions = [p.replace('\\', '/') for p in suggestions]

    result = {
        "suggestions": suggestions
    }

    return json.dumps(result)

@app.route('/workspaces', methods=['GET'])
def show_workspaces():
    """
    Lists the workspaces in DIGITS
    """
    jobs_dir = config_value('jobs_dir')
    workspaces = next(os.walk(jobs_dir))[1]
    print workspaces
    return render_template('workspaces.html', workspaces = workspaces)

@app.route('/workspaces', methods=['POST'])
def create_workspaces():
    """
    Creates a new Workspace in DIGITS
    """
    error = False
    new_workspace=request.form['name']
    jobs_dir = config_value('jobs_dir')
    if new_workspace in next(os.walk(jobs_dir))[1]:
        error  = 'Workspace with this name already exists! Please choose another name.'
    else:
        _dir = os.path.join(config_value('jobs_dir'), new_workspace)
        os.mkdir(_dir)
    workspaces = next(os.walk(jobs_dir))[1]
    return render_template('workspaces.html', workspaces = workspaces, error = error)

@app.route('/workspace/edit/<workspace>', methods=['POST'])
def edit_workspace_name(workspace):
    """
    Edit the name of a Workspace
    """

    jobs_dir = config_value('jobs_dir')
    workspaces = next(os.walk(jobs_dir))[1]
    old_workspace_name = workspace
    new_workspace_name = request.form['workspace_name']
    if old_workspace_name == "Default":
        return render_template('workspaces.html', workspaces = workspaces, error = "Cannot modify the Default workspace of DIGITS")
    elif not os.path.exists(jobs_dir+"/"+new_workspace_name):
        os.rename(jobs_dir+"/"+old_workspace_name, jobs_dir+"/"+new_workspace_name)
    else:
        return render_template('workspaces.html', workspaces = workspaces, error = "A workspace already exist with this name")
    workspaces = next(os.walk(jobs_dir))[1]
    return render_template('workspaces.html', workspaces = workspaces, error = None)

@app.route('/workspace/delete', methods = ['DELETE'])
def delete_workspace():
    workspace = get_workspace(flask.request.url)
    jobs_dir = config_value('jobs_dir')
    workspaces = next(os.walk(jobs_dir))[1]
    if workspace == "Default":
        return render_template('workspaces.html', workspaces = workspaces, error = "Cannot Delete the Default workspace of DIGITS")
    else:
        shutil.rmtree(jobs_dir+'/'+workspace)
        return render_template('workspaces.html', workspaces = workspaces, success = "Successfully deleted the workspace %s" % workspace)

### SocketIO functions

## /home

@socketio.on('connect', namespace='/home')
def on_connect_home():
    """
    Somebody connected to the homepage
    """
    pass

@socketio.on('disconnect', namespace='/home')
def on_disconnect_home():
    """
    Somebody disconnected from the homepage
    """
    pass

## /jobs

@socketio.on('connect', namespace='/jobs')
def on_connect_jobs():
    """
    Somebody connected to a jobs page
    """
    pass

@socketio.on('disconnect', namespace='/jobs')
def on_disconnect_jobs():
    """
    Somebody disconnected from a jobs page
    """
    pass

@socketio.on('join', namespace='/jobs')
def on_join_jobs(data):
    """
    Somebody joined a room
    """
    room = data['room']
    join_room(room)
    flask.session['room'] = room

@socketio.on('leave', namespace='/jobs')
def on_leave_jobs():
    """
    Somebody left a room
    """
    if 'room' in flask.session:
        room = flask.session['room']
        del flask.session['room']
        #print '>>> Somebody left room %s' % room
        leave_room(room)

