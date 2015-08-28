# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import flask
from flask.ext.socketio import SocketIO

from digits import utils
from config import config_value
import digits.scheduler

### Create Flask, Scheduler and SocketIO objects

app = flask.Flask(__name__)
app.config['DEBUG'] = True
# Disable CSRF checking in WTForms
app.config['WTF_CSRF_ENABLED'] = False
# This is still necessary for SocketIO
app.config['SECRET_KEY'] = config_value('secret_key')
app.url_map.redirect_defaults = False
socketio = SocketIO(app)
scheduler = digits.scheduler.Scheduler(config_value('gpu_list'))

# Set up flask API documentation, if installed
try:
    from flask.ext.autodoc import Autodoc
    _doc = Autodoc(app)
    autodoc = _doc.doc # decorator
except ImportError:
    def autodoc(*args, **kwargs):
        def _doc(f):
            # noop decorator
            return f
        return _doc

### Register filters and views

app.jinja_env.globals['server_name'] = config_value('server_name')
app.jinja_env.globals['server_version'] = digits.__version__
app.jinja_env.filters['print_time'] = utils.time_filters.print_time
app.jinja_env.filters['print_time_diff'] = utils.time_filters.print_time_diff
app.jinja_env.filters['print_time_since'] = utils.time_filters.print_time_since
app.jinja_env.filters['sizeof_fmt'] = utils.sizeof_fmt
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True

import digits.views

### Setup the environment

scheduler.load_past_jobs()
