# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import json
import os
import requests
import tempfile
import uuid

import flask

from digits.log import logger
from digits.pretrained_model import PretrainedModelJob
from digits.utils import auth
from digits.utils.store import StoreParser
from digits.webapp import app, scheduler

blueprint = flask.Blueprint(__name__, __name__)

def save_binary(url, file_name, tmp_dir):
    r = requests.get(os.path.join(url, file_name), stream=True)
    full_path = os.path.join(tmp_dir, file_name)
    with open(full_path,'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    return full_path

def retrieve_files(url, directory):
    model_url = os.path.join(url, directory)
    tmp_dir = tempfile.mkdtemp()
    info = json.loads(requests.get(os.path.join(model_url, 'info.json')).content)
    weights = save_binary(model_url, info["snapshot file"], tmp_dir)
    if "model file" in info:
        remote_model_file = info["model file"]
    elif "network file" in info:
        remote_model_file = info["network file"]
    else:
        return flask.jsonify({"status": "Missing model definition in info.json"}), 500
    model = save_binary(model_url, remote_model_file, tmp_dir)
    if "labels file" in info:
        label = save_binary(model_url, info["labels file"], tmp_dir)
    else:
        label = None
    meta_data = info
    return weights, model, label, meta_data

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
        return 'Error', 404
    else:
        weights, model, label, meta_data = retrieve_files(url, directory)
        job = PretrainedModelJob(
            weights,
            model,
            label,
            meta_data['framework'],
            username=auth.get_username(),
            name=meta_data['name']
        )
        scheduler.add_job(job)
        return flask.redirect(flask.url_for('digits.views.home', tab=3)), 302

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
            page = requests.get(store_base_url)
        except requests.exceptions.RequestException as e:
            logger.warning('Skip %s due to error %s' % (store_base_url, e))
            continue
        parser = StoreParser()
        parser.feed(page.content)
        msg = 'Thanks for visiting {}'.format(store_base_url)
        if len(parser.get_child_dirs()) > 0:
            dirs = [d[:-1] for d in parser.get_child_dirs()]
        else:
            response = requests.get(os.path.join(store_base_url,'master.json'))
            if response.status_code == 200:
                json_response = json.loads(response.content)
                dirs = json_response['children']
                msg = json_response['msg']
            else:
                continue
        for subdir in dirs:
            tmp_dict = {'dir_name': subdir}
            response = requests.get(os.path.join(store_base_url,subdir,'info.json'))
            if response.status_code == 200:
                tmp_dict['info'] = json.loads(response.content)
                tmp_dict['id'] = str(uuid.uuid4())
            response = requests.get(os.path.join(store_base_url,subdir,'aux.json'))
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
