# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import os.path
import pickle
import shutil
import threading
import time

import flask

from .status import Status, StatusCls
from digits.config import config_value
from digits.utils import sizeof_fmt, filesystem as fs

# NOTE: Increment this every time the pickled object changes
PICKLE_VERSION = 2


class Job(StatusCls):
    """
    Base class
    """
    SAVE_FILE = 'status.pickle'

    @classmethod
    def load(cls, job_id):
        """
        Loads a Job in the given job_id
        Returns the Job or throws an exception
        """
        from digits.model.tasks import TrainTask

        job_dir = os.path.join(config_value('jobs_dir'), job_id)
        filename = os.path.join(job_dir, cls.SAVE_FILE)
        with open(filename, 'rb') as savefile:
            job = pickle.load(savefile)
            # Reset this on load
            job._dir = job_dir
            for task in job.tasks:
                task.job_dir = job_dir
                if isinstance(task, TrainTask):
                    # can't call this until the job_dir is set
                    task.detect_snapshots()
            return job

    def __init__(self, name, username, group='', persistent=True):
        """
        Arguments:
        name -- name of this job
        username -- creator of this job
        """
        super(Job, self).__init__()

        # create a unique ID
        self._id = '%s-%s' % (time.strftime('%Y%m%d-%H%M%S'), os.urandom(2).encode('hex'))
        self._dir = os.path.join(config_value('jobs_dir'), self._id)
        self._name = name
        self.group = group
        self.username = username
        self.pickver_job = PICKLE_VERSION
        self.tasks = []
        self.exception = None
        self._notes = None
        self.event = threading.Event()
        self.persistent = persistent

        os.mkdir(self._dir)

    def __getstate__(self):
        """
        Used when saving a pickle file
        """
        d = self.__dict__.copy()
        # Isn't linked to state
        if '_dir' in d:
            del d['_dir']
        if 'event' in d:
            del d['event']

        return d

    def __setstate__(self, state):
        """
        Used when loading a pickle file
        """
        if 'username' not in state:
            state['username'] = None
        if 'group' not in state:
            state['group'] = ''
        self.__dict__ = state
        self.persistent = True

    def json_dict(self, detailed=False):
        """
        Returns a dict used for a JSON representation
        """
        d = {
            'id': self.id(),
            'name': self.name(),
            'status': self.status.name,
        }
        if detailed:
            d.update({
                'directory': self.dir(),
            })
        return d

    def id(self):
        """getter for _id"""
        return self._id

    def dir(self):
        """getter for _dir"""
        return self._dir

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
            path = os.path.join(self._dir, filename)
            if relative:
                path = os.path.relpath(path, config_value('jobs_dir'))
        return str(path).replace("\\", "/")

    def path_is_local(self, path):
        """assert that a path is local to _dir"""
        p = os.path.normpath(path)
        if os.path.isabs(p):
            return False
        if p.startswith('..'):
            return False
        return True

    def name(self):
        return self._name

    def notes(self):
        if hasattr(self, '_notes'):
            return self._notes
        else:
            return None

    def job_type(self):
        """
        String representation for this class
        virtual function
        """
        raise NotImplementedError('Implement me!')

    def status_of_tasks(self):
        """
        Returns the status of the job's tasks
        """
        job_status = self.status

        if job_status in [Status.ABORT, Status.ERROR]:
            return job_status

        task_statuses = [t.status for t in self.tasks]

        important_statuses = [
            Status(s) for s in [
                # Sorted by importance
                Status.ERROR,
                Status.ABORT,
                Status.RUN,
                Status.WAIT,
                Status.INIT,
                Status.DONE,
            ]
        ]
        for s in important_statuses:
            # Return if any task matches
            if s in task_statuses:
                return s

        return Status(Status.DONE)

    def runtime_of_tasks(self):
        """
        Returns the time (in sec) between when the first task started and when the last task stopped
        NOTE: this may not be what you're expecting if there was some WAIT time in-between
        """
        starts = []
        stops = []
        for task in self.tasks:
            for start_status, start_timestamp in task.status_history:
                if start_status == Status.RUN:
                    starts.append(start_timestamp)
                    # Only search for stops if the task was started at some point
                    for stop_status, stop_timestamp in task.status_history:
                        if stop_status in [Status.DONE, Status.ABORT, Status.ERROR]:
                            stops.append(stop_timestamp)
                    break

        if len(starts):
            min_start = min(starts)
            if len(stops):
                max_stop = max(stops)
                return max_stop - min_start
            else:
                return time.time() - min_start
        else:
            return 0

    def on_status_update(self):
        """
        Called when StatusCls.status.setter is used
        """
        from digits.webapp import app, socketio

        message = {
            'update': 'status',
            'status': self.status_of_tasks().name,
            'css': self.status_of_tasks().css,
            'running': self.status.is_running(),
            'job_id': self.id(),
        }
        with app.app_context():
            message['html'] = flask.render_template('status_updates.html', updates=self.status_history)

        socketio.emit('job update',
                      message,
                      namespace='/jobs',
                      room=self.id(),
                      )

        # send message to job_management room as well
        socketio.emit('job update',
                      message,
                      namespace='/jobs',
                      room='job_management',
                      )

        if not self.status.is_running():
            if hasattr(self, 'event'):
                # release threads that are waiting for job to complete
                self.event.set()

    def abort(self):
        """
        Abort a job and stop all running tasks
        """
        if self.status.is_running():
            self.status = Status.ABORT
        for task in self.tasks:
            task.abort()

    def save(self):
        """
        Saves the job to disk as a pickle file
        Suppresses errors, but returns False if something goes wrong
        """
        try:
            # use tmpfile so we don't abort during pickle dump (leading to EOFErrors)
            tmpfile_path = self.path(self.SAVE_FILE + '.tmp')
            with open(tmpfile_path, 'wb') as tmpfile:
                pickle.dump(self, tmpfile)
            shutil.move(tmpfile_path, self.path(self.SAVE_FILE))
            return True
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print 'Caught %s while saving job %s: %s' % (type(e).__name__, self.id(), e)
        return False

    def disk_size_fmt(self):
        """
        return string representing job disk size
        """
        size = fs.get_tree_size(self._dir)
        return sizeof_fmt(size)

    def get_progress(self):
        """
        Return job progress computed from task progress
        """
        if len(self.tasks) == 0:
            return 0.0

        progress = 0.0

        for task in self.tasks:
            progress += task.progress

        progress /= len(self.tasks)
        return progress

    def emit_progress_update(self):
        """
        Call socketio.emit for task job update, by considering task progress.
        """
        progress = self.get_progress()

        from digits.webapp import socketio
        socketio.emit('job update',
                      {
                          'job_id': self.id(),
                          'update': 'progress',
                          'percentage': int(round(100 * progress)),
                      },
                      namespace='/jobs',
                      room='job_management'
                      )

    def emit_attribute_changed(self, attribute, value):
        """
        Call socketio.emit for task job update
        """
        from digits.webapp import socketio
        socketio.emit('job update',
                      {
                          'job_id': self.id(),
                          'update': 'attribute',
                          'attribute': attribute,
                          'value': value,
                      },
                      namespace='/jobs',
                      room='job_management'
                      )

    def wait_completion(self):
        """
        Wait for the job to complete
        """
        # if job was loaded from disk (which is the only case
        # when the 'event' attribute should be missing) then
        # assume it has completed already (done, errored or interrupted)
        if hasattr(self, 'event'):
            self.event.wait()

    def is_persistent(self):
        """
        Returns whether job is persistent
        """
        return self.persistent

    def is_read_only(self):
        """
        Returns False if this job can be edited
        """
        return not self.is_persistent()
