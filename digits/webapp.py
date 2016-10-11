# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

import flask
from flask.ext.socketio import SocketIO
from gevent import monkey; monkey.patch_all()

from .config import config_value
from digits import utils
from digits.utils import filesystem as fs
from digits.utils.store import StoreCache
import digits.scheduler

### Create Flask, Scheduler and SocketIO objects

app = flask.Flask(__name__)
app.config['DEBUG'] = True
# Disable CSRF checking in WTForms
app.config['WTF_CSRF_ENABLED'] = False
# This is still necessary for SocketIO
app.config['SECRET_KEY'] = os.urandom(12).encode('hex')
app.url_map.redirect_defaults = False
socketio = SocketIO(app, async_mode='gevent')
app.config['store_cache'] = StoreCache()
app.config['store_url_list'] = config_value('model_store')['url_list']
scheduler = digits.scheduler.Scheduler(config_value('gpu_list'), True)

### Register filters and views

app.jinja_env.globals['server_name'] = config_value('server_name')
app.jinja_env.globals['server_version'] = digits.__version__
app.jinja_env.globals['caffe_version'] = config_value('caffe')['version']
app.jinja_env.globals['caffe_flavor'] = config_value('caffe')['flavor']
app.jinja_env.globals['dir_hash'] = fs.dir_hash(
    os.path.join(os.path.dirname(digits.__file__), 'static'))
app.jinja_env.filters['print_time'] = utils.time_filters.print_time
app.jinja_env.filters['print_time_diff'] = utils.time_filters.print_time_diff
app.jinja_env.filters['print_time_since'] = utils.time_filters.print_time_since
app.jinja_env.filters['sizeof_fmt'] = utils.sizeof_fmt
app.jinja_env.filters['has_permission'] = utils.auth.has_permission
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True

import digits.views
app.register_blueprint(digits.views.blueprint)
import digits.dataset.views
app.register_blueprint(digits.dataset.views.blueprint, url_prefix='/datasets')
import digits.dataset.generic.views
app.register_blueprint(digits.dataset.generic.views.blueprint, url_prefix='/datasets/generic')
import digits.dataset.images.views
app.register_blueprint(digits.dataset.images.views.blueprint, url_prefix='/datasets/images')
import digits.dataset.images.classification.views
app.register_blueprint(digits.dataset.images.classification.views.blueprint, url_prefix='/datasets/images/classification')
import digits.dataset.images.generic.views
app.register_blueprint(digits.dataset.images.generic.views.blueprint, url_prefix='/datasets/images/generic')
import digits.model.views
app.register_blueprint(digits.model.views.blueprint, url_prefix='/models')
import digits.model.images.views
app.register_blueprint(digits.model.images.views.blueprint, url_prefix='/models/images')
import digits.model.images.classification.views
app.register_blueprint(digits.model.images.classification.views.blueprint, url_prefix='/models/images/classification')
import digits.model.images.generic.views
app.register_blueprint(digits.model.images.generic.views.blueprint, url_prefix='/models/images/generic')
import digits.pretrained_model.views
app.register_blueprint(digits.pretrained_model.views.blueprint, url_prefix='/pretrained_models')
import digits.store.views
app.register_blueprint(digits.store.views.blueprint, url_prefix='/store')

def username_decorator(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        this_username = flask.request.cookies.get('username', None)
        app.jinja_env.globals['username'] = this_username
        return f(*args, **kwargs)
    return decorated

for endpoint, function in app.view_functions.iteritems():
    app.view_functions[endpoint] = username_decorator(function)

### Setup the environment

scheduler.load_past_jobs()
