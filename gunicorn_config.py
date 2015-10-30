# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

worker_class = 'socketio.sgunicorn.GeventSocketIOWorker'
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
