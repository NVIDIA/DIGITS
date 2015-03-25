# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path
import re
import time
import logging
import subprocess

from flask import render_template
import gevent.event

from . import utils
import digits.log
from config import config_option
from status import Status, StatusCls

# NOTE: Increment this everytime the pickled version changes
PICKLE_VERSION = 1

class Task(StatusCls):
    """
    Base class for Tasks
    A Task is a compute-heavy operation that runs in a separate executable
    Communication is done by processing the stdout of the executable
    """

    def __init__(self, job_dir, parents=None):
        super(Task, self).__init__()
        self.pickver_task = PICKLE_VERSION

        self.job_dir = job_dir
        self.job_id = os.path.basename(job_dir)

        if parents is None:
            self.parents = None
        elif isinstance(parents, (list, tuple)):
            self.parents = parents
        elif isinstance(parents, Task):
            self.parents = [parents]
        else:
            raise TypeError('parents is %s' % type(parents))

        self.progress = 0
        self.exception = None
        self.traceback = None
        self.aborted = gevent.event.Event()
        self.set_logger()

    def __getstate__(self):
        d = self.__dict__.copy()

        if 'aborted' in d:
            del d['aborted']
        if 'logger' in d:
            del d['logger']

        return d

    def __setstate__(self, state):
        self.__dict__ = state

        self.aborted = gevent.event.Event()
        self.set_logger()

    def set_logger(self):
        self.logger = digits.log.JobIdLoggerAdapter(
                logging.getLogger('digits.webapp'),
                {'job_id': self.job_id},
                )

    def name(self):
        """
        Returns a string
        """
        raise NotImplementedError('Please implement me')

    def html_id(self):
        """
        Returns a string
        """
        return 'task-%s' % id(self)

    def on_status_update(self):
        """
        Called when StatusCls.status.setter is used
        """
        from digits.webapp import app, socketio

        # Send socketio updates
        message = {
                'task': self.html_id(),
                'update': 'status',
                'status': self.status.name,
                'css': self.status.css,
                'show': (self.status in [Status.RUN, Status.ERROR]),
                'running': self.status.is_running(),
                }
        with app.app_context():
            message['html'] = render_template('status_updates.html',
                    updates     = self.status_history,
                    exception   = self.exception,
                    traceback   = self.traceback,
                    )

        socketio.emit('task update',
                message,
                namespace='/jobs',
                room=self.job_id,
                )

    def path(self, filename, relative=False):
        """
        Returns a path to the given file

        Arguments:
        filename -- the requested file

        Keyword arguments:
        relative -- If False, return an absolute path to the file
                    If True, return a path relative to the jobs directory
        """
        if not filename:
            return None
        if os.path.isabs(filename):
            path = filename
        else:
            path = os.path.join(self.job_dir, filename)
        if relative:
            path = os.path.relpath(path, config_option('jobs_dir'))
        return str(path)

    def ready_to_queue(self):
        """
        Returns True if all parents are done
        """
        if not self.parents:
            return True
        for parent in self.parents:
            if parent.status != Status.DONE:
                return False
        return True

    def task_arguments(self, **kwargs):
        """
        Returns args used by subprocess.Popen to execute the task
        Returns False if the args cannot be set properly
        """
        raise NotImplementedError('Please implement me')

    def before_run(self):
        """
        Called before run() executes
        Raises exceptions
        """
        pass

    def run(self, **kwargs):
        """
        Execute the task
        """
        self.before_run()

        args = self.task_arguments(**kwargs)
        if not args:
            self.logger.error('Could not create the arguments for Popen')
            self.status = Status.ERROR
            return False
        # Convert them all to strings
        args = [str(x) for x in args]

        self.logger.info('%s task started.' % self.name())
        self.status = Status.RUN

        unrecognized_output = []

        p = subprocess.Popen(args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.job_dir,
                close_fds=True,
                )

        try:
            while p.poll() is None:
                for line in utils.nonblocking_readlines(p.stdout):
                    if self.aborted.is_set():
                        p.terminate()
                        self.status = Status.ABORT
                        break

                    if line is not None:
                        # Remove whitespace
                        line = line.strip()

                    if line:
                        if not self.process_output(line):
                            self.logger.warning('%s unrecognized input: %s' % (self.name(), line.strip()))
                            unrecognized_output.append(line)
                    else:
                        time.sleep(0.05)
        except:
            p.terminate()
            self.after_run()
            raise

        self.after_run()

        if self.status != Status.RUN:
            return False
        elif p.returncode != 0:
            self.logger.error('%s task failed with error code %d' % (self.name(), p.returncode))
            if self.exception is None:
                self.exception = 'error code %d' % p.returncode
                if unrecognized_output:
                    self.traceback = '\n'.join(unrecognized_output)
            self.after_runtime_error()
            self.status = Status.ERROR
            return False
        else:
            self.logger.info('%s task completed.' % self.name())
            self.status = Status.DONE
            return True

    def abort(self):
        """
        Abort the Task
        """
        if self.status.is_running():
            self.aborted.set()

    def preprocess_output_digits(self, line):
        """
        Takes line of output and parses it according to DiGiTS's log format
        Returns (timestamp, level, message) or (None, None, None)
        """
        # NOTE: This must change when the logging format changes
        # YYYY-MM-DD HH:MM:SS [LEVEL] message
        match = re.match(r'(\S{10} \S{8}) \[(\w+)\s*\] (.*)$', line)
        if match:
            timestr = match.group(1)
            timestamp = time.mktime(time.strptime(timestr, digits.log.DATE_FORMAT))
            level = match.group(2)
            message = match.group(3)
            if level.startswith('DEB'):
                level = 'debug'
            elif level.startswith('INF'):
                level = 'info'
            elif level.startswith('WAR'):
                level = 'warning'
            elif level.startswith('ERR'):
                level = 'error'
            elif level.startswith('CRI'):
                level = 'critical'
            return (timestamp, level, message)
        else:
            return (None, None, None)


    def process_output(self, line):
        """
        Process a line of output from the task
        Returns True if the output was able to be processed

        Arguments:
        line -- a line of output
        """
        raise NotImplementedError('Please implement me')

    def est_done(self):
        """
        Returns the estimated time in seconds until the task is done
        """
        if self.status != Status.RUN or self.progress == 0:
            return None
        elapsed = time.time() - self.status_history[-1][1]
        return (1 - self.progress) * elapsed // self.progress

    def after_run(self):
        """
        Called after run() executes
        """
        pass

    def after_runtime_error(self):
        """
        Called after a runtime error during run()
        """
        pass

