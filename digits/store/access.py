#!/usr/bin/env python2
# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.

import argparse
import json
import os
import tarfile
import zipfile

from flask import Flask, request, redirect, url_for, make_response
from flask.ext.cors import CORS

from store_sqlalchemy_backend import ModelStore

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['caffemodel', 'prototxt'])

app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = 'We need a secret key here.'
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['model_store'] = ModelStore('test.sqlite3')
cors = CORS(app)

mapping_dict = {"train_val.prototxt": "train_val_file",
                "deploy.prototxt": "deploy_file",
                "solver.prototxt": "solver_file",
                "labels.txt": "label_file",
                "mean.binaryproto": "mean_file",
                "info.json": "info_file",
                "original.prototxt": "network_file"}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/store/file/<file_key>')
def download_one_file(file_key):
    file_name, file_content = app.config['model_store'].get_file(file_key)
    response = make_response(file_content)
    response.headers["Content-Disposition"] = "attachment; filename={}".format(file_name)
    response.mimetype = "application/octet-stream"
    return response


@app.route('/store/model/<model_key>', methods=['GET', 'DELETE'])
def handle_one_model(model_key):
    if request.method == 'GET':
        zf, name = app.config['model_store'].get_one(model_key)
        response = make_response(zf.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename={}.zip".format(name)
        response.mimetype = "application/octet-stream"
        return response
    elif request.method == 'DELETE':
        result = app.config['model_store'].delete_one(model_key)
        return '200', 'OK'


@app.route('/store/models')
def download_all_models():
    v = app.config['model_store'].get_zip()
    response = make_response(v.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=models.zip"
    response.mimetype = "application/octet-stream"
    return response


@app.route('/uploads/<model_key>.json')
def uploaded_file(model_key):
    mr = app.config['model_store'].get_model(model_key)
    return json.dumps(mr)


@app.route('/store')
def list_models():
    model_list = app.config['model_store'].models()
    return json.dumps(model_list)


@app.route('/store/model/crud', methods=['GET', 'POST'])
def model_crud():
    if request.method == 'POST':
        model_keys = request.form['model_keys']
    else:
        model_keys = request.args.getlist('model_keys')
    if 'delete' in request.args:
        for model_key in model_keys:
            app.config['model_store'].delete_one(model_key)
        app.config['model_store'].commit()
        return redirect(url_for('publish_model'))
    elif 'download' in request.args:
        v = app.config['model_store'].get_models_zip(model_keys)
        response = make_response(v.getvalue())
        response.headers["Content-Disposition"] = "attachment; filename=models.zip"
        response.mimetype = "application/octet-stream"
        return response
    return redirect(url_for('publish_model'))


@app.route('/publish', methods=['GET', 'POST'])
def publish_model():
    if request.method == 'POST':
        received_files = request.files
        if len(received_files['archive'].filename)>0:
            files = list()
            archive_file = received_files['archive']
            _, extension = os.path.splitext(archive_file.filename)
            if extension in ['.zip']:
                zf = zipfile.ZipFile(archive_file)
                for name in zf.namelist():
                    file_dict = dict()
                    if name.endswith('.caffemodel'):
                        file_dict['form_name'] = 'weight_file'
                    else:
                        file_dict['form_name'] = mapping_dict.get(name, name)
                    file_dict['file_name'] = name
                    file_dict['content'] = zf.read(name)
                    files.append(file_dict)
            elif extension in ['.tar', '.gzip', '.gz', '.tgz']:
                tar = tarfile.open(fileobj=archive_file.stream)
                for tar_info in tar.getmembers():
                    file_dict = dict()
                    if tar_info.name.endswith('.caffemodel'):
                        file_dict['form_name'] = 'weight_file'
                    else:
                        file_dict['form_name'] = mapping_dict.get(tar_info.name, tar_info.name)
                    file_dict['file_name'] = tar_info.name
                    file_dict['content'] = tar.extractfile(tar_info).read()
                    files.append(file_dict)
        else:
            files = list()
            for name in received_files:
                if name != 'archive':
                    file_dict = dict()
                    file_dict['form_name'] = name
                    file_dict['file_name'] = received_files[name].filename
                    file_dict['content'] = received_files[name].read()
                    files.append(file_dict)
            info = {
                'snapshot file': received_files['weight_file'].filename,
                'model file': received_files['network_file'].filename,
                'user': request.form['model_user']
            }
            files.append({'form_name': 'info.json',
                          'file_name': 'info.json',
                          'content': json.dumps(info)})
        model = {'name': request.form['model_name'],
                 'description': request.form['description'],
                 'instruction': request.form['instruction'],
                 'dataset': request.form['dataset'],
                 'license': request.form['license'],
                 'files': files}
        model_key = app.config['model_store'].add(model)
        #return redirect(url_for('uploaded_file', model_key=model_key))
    return redirect(url_for('manage_model_store'))


@app.route('/')
def manage_model_store():
    return app.send_static_file('home.html')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Store server')
    parser.add_argument('-p', '--port',
            type=int,
            default=5050,
            help='Port to run app on (default 5050)'
            )
    args = vars(parser.parse_args())
    app.run(host='0.0.0.0', port=args['port'], threaded=True)
