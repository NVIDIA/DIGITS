# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import sys

worker_class = 'socketio.sgunicorn.GeventSocketIOWorker'
bind = '0.0.0.0:8080'
loglevel = 'debug'

def on_starting(server):
    from digits import config
    if not config.valid_config():
        sys.exit(1)

def post_fork(server, worker):
    from digits.webapp import scheduler
    scheduler.start()

def worker_exit(server, worker):
    from digits.webapp import scheduler
    scheduler.stop()
