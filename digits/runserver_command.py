# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
"""
Flask-Script runserver command
"""
from __future__ import absolute_import

import sys

from flask.ext.script import Command, Option
from flask.ext.socketio import SocketIO

from digits.webapp import app, socketio, scheduler


class ServerCommand(Command):
    """
    Patterned after flask.ext.script.Server
    """
    help = description = 'Runs the Flask-SocketIO development server'

    def __init__(self, host='0.0.0.0', port=5000, use_debugger=True,
                 use_reloader=False, threaded=False, processes=1,
                 passthrough_errors=False, **options):

        self.port = port
        self.host = host
        self.use_debugger = use_debugger
        self.use_reloader = use_reloader
        self.server_options = options
        self.threaded = threaded
        self.processes = processes
        self.passthrough_errors = passthrough_errors

    def get_options(self):

        options = (
            Option('-t', '--host',
                   dest='host',
                   default=self.host),

            Option('-p', '--port',
                   dest='port',
                   type=int,
                   default=self.port),
        )

        if self.use_debugger:
            options += (Option('-d', '--no-debug',
                               action='store_false',
                               dest='use_debugger',
                               default=self.use_debugger),)

        else:
            options += (Option('-d', '--debug',
                               action='store_true',
                               dest='use_debugger',
                               default=self.use_debugger),)

        if self.use_reloader:
            options += (Option('-r', '--no-reload',
                               action='store_false',
                               dest='use_reloader',
                               default=self.use_reloader),)

        else:
            options += (Option('-r', '--reload',
                               action='store_true',
                               dest='use_reloader',
                               default=self.use_reloader),)

        return options

    def handle(self, app, host, port, use_debugger, use_reloader):
        try:
            if not scheduler.start():
                print 'ERROR: Scheduler would not satart'
            else:
                app.debug = use_debugger
                socketio.run(app,
                             host=host,
                             port=port,
                             use_reloader=use_reloader,
                             policy_server=False,
                             )
        except KeyboardInterrupt:
            pass
        finally:
            scheduler.stop()
