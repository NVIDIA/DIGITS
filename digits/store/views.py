# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import json
import os
import tempfile
import time
import uuid

import flask
import requests

from digits.log import logger
from digits.pretrained_model import PretrainedModelJob
from digits.utils import auth
from digits.utils.store import StoreParser
from digits.webapp import app, scheduler, socketio


blueprint = flask.Blueprint(__name__, __name__)


class Progress(object):
    """class to emit download progress"""

    def __init__(self, model_id):
        self._model_id = model_id
        self._file = 0
        self._n_files = 0
        self._n_chunks = 0
        self._last_progress = -1
        self.emit(0)

    def set_n_files(self, n_files):
        """ set the number of files file this Progress object will report """
        self._n_files = n_files

    def set_n_chunks(self, n_chuncks):
        """ set the number of chunks expected """
        self._n_chunks = n_chuncks
        self._file += 1
        self._last_progress = -1

    def emit(self, progress):
        """ emit the progress to the client """
        socketio.emit('update',
                      {
                          'model_id': self._model_id,
                          'update': 'progress',
                          'progress': progress,
                      },
                      namespace='/jobs',
                      room='job_management'
                      )
        # micro sleep so that emit is broadcast to the client
        time.sleep(0.001)

    def incr(self, itr):
        """ progress iterator that the request iterator is wrapped in """
        for i, item in enumerate(itr):
            yield item
            progress = min(int(round(((self._file - 1.0) + (i + 1.0) / self._n_chunks) /
                                     self._n_files * 100)), 100)
            if progress != self._last_progress:
                self.emit(progress)
                self._last_progress = progress


def save_binary(url, file_name, tmp_dir, progress):
    r = requests.get(os.path.join(url, file_name), stream=True)
    chunk_size = 1024
    total_length = int(r.headers.get('content-length'))
    n_chuncks = (total_length / chunk_size) + bool(total_length % chunk_size)
    progress.set_n_chunks(n_chuncks)
    full_path = os.path.join(tmp_dir, file_name)
    with open(full_path, 'wb') as f:
        for chunk in progress.incr(r.iter_content(chunk_size=chunk_size)):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    return full_path


def retrieve_files(url, directory, progress):
    model_url = os.path.join(url, directory)
    tmp_dir = tempfile.mkdtemp()
    info = json.loads(requests.get(os.path.join(model_url, 'info.json')).content)

    # How many files will we download?
    n_files = 1 + ("model file" in info or "network file" in info) + ("labels file" in info)
    progress.set_n_files(n_files)

    weights = save_binary(model_url, info["snapshot file"], tmp_dir, progress)
    if "model file" in info:
        remote_model_file = info["model file"]
    elif "network file" in info:
        remote_model_file = info["network file"]
    else:
        return flask.jsonify({"status": "Missing model definition in info.json"}), 500
    model = save_binary(model_url, remote_model_file, tmp_dir, progress)
    if "labels file" in info:
        label = save_binary(model_url, info["labels file"], tmp_dir, progress)
    else:
        label = None
    if "python layer file" in info:
        python_layer = save_binary(model_url, info["python layer file"], tmp_dir, progress)
    else:
        python_layer = None
    meta_data = info
    return weights, model, label, meta_data, python_layer


@blueprint.route('/push', methods=['GET'])
def push():
    """
    Create a pre-trained model from model store
    """
    model_id = flask.request.args.get('id')
    model_grand_list = app.config['store_cache'].read()
    found = False
    if model_grand_list is not None:
        for store in model_grand_list:
            for model in model_grand_list[store]['model_list']:
                if model['id'] == model_id:
                    url = model_grand_list[store]['base_url']
                    directory = model['dir_name']
                    found = True
                    break
            if found:
                break
    if not found:
        return 'Unable to find requested model', 404
    else:
        progress = Progress(model_id)
        weights, model, label, meta_data, python_layer = retrieve_files(url, directory, progress)
        job = PretrainedModelJob(
            weights,
            model,
            label,
            meta_data['framework'],
            username=auth.get_username(),
            name=meta_data['name']
        )
        scheduler.add_job(job)
        response = flask.make_response(job.id())
        return response


@blueprint.route('/models', methods=['GET'])
def models():
    """
    perform server-to-server communication to retrieve
    info.json file in all subfolders of model store
    """
    if flask.request.args.get('refresh') == '1':
        app.config['store_cache'].reset()
    cached_data = app.config['store_cache'].read()
    if cached_data is not None:
        return json.dumps(cached_data)

    store_urls = app.config['store_url_list']
    aggregated_dict = dict()
    for i, store_url in enumerate(store_urls):
        if len(store_url) == 0:
            continue
        model_list = list()
        if store_url[-1] != '/':
            store_base_url = store_url + '/'
        else:
            store_base_url = store_url

        try:
            response = requests.get(os.path.join(store_base_url, 'master.json'))
            if response.status_code == 200:
                json_response = json.loads(response.content)
                dirs = json_response['children']
                msg = json_response['msg']
            else:  # try to retrieve from directory listing
                page = requests.get(store_base_url)
                parser = StoreParser()
                parser.feed(page.content)
                if len(parser.get_child_dirs()) > 0:  # we have list of subdirectories
                    dirs = [d[:-1] for d in parser.get_child_dirs()]
                    msg = 'Thanks for visiting {}'.format(store_base_url)
                else:  # nothing found, try next URL
                    continue
        except requests.exceptions.RequestException as e:
            logger.warning('Skip %s due to error %s' % (store_base_url, e))
            continue

        for subdir in dirs:
            tmp_dict = {'dir_name': subdir}
            response = requests.get(os.path.join(store_base_url, subdir, 'info.json'))
            if response.status_code == 200:
                tmp_dict['info'] = json.loads(response.content)
                tmp_dict['id'] = str(uuid.uuid4())
            response = requests.get(os.path.join(store_base_url, subdir, 'aux.json'))
            if response.status_code == 200:
                tmp_dict['aux'] = json.loads(response.content)
            model_list.append(tmp_dict)
        store_info = {'base_url': store_base_url, 'welcome_msg': msg,
                      'model_list': model_list}
        aggregated_dict[store_base_url] = store_info
    app.config['store_cache'].write(aggregated_dict)
    return json.dumps(aggregated_dict)


@blueprint.route('/list', methods=['GET'])
def store():
    return flask.render_template(
        'store.html'
    )
