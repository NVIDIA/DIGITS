# Copyright (c) 2014-2016, NVIDIA CORPORATION.  All rights reserved.

import os.path
import sys

# Add path to digits module
sys.path.append(os.path.dirname(__file__))

worker_class = 'geventwebsocket.gunicorn.workers.GeventWebSocketWorker'
bind = '0.0.0.0:34448' # DIGIT
loglevel = 'debug'

def on_starting(server):
    from digits import config
    config.load_config()

def post_fork(server, worker):
    from digits.webapp import scheduler
    scheduler.start()

def worker_exit(server, worker):
    from digits.webapp import scheduler
    scheduler.stop()
