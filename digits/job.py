# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import time
import os.path
import time
import pickle
import shutil

from flask import render_template

from digits import utils
from digits.config import config_option
from status import Status, StatusCls

# NOTE: Increment this everytime the pickled object changes
PICKLE_VERSION = 1

class Job(StatusCls):
    """
    Base class
    """
    SAVE_FILE = 'status.pickle'

    @classmethod
    def load(cls, job_id):
        """
        Loads a Job in the given job_id
        Returns the Job or None if an error occurred
        """
        job_dir = os.path.join(config_option('jobs_dir'), job_id)
        filename = os.path.join(job_dir, cls.SAVE_FILE)
        with open(filename, 'rb') as savefile:
            o = pickle.load(savefile)
            # Reset this on load
            o._dir = job_dir
            return o

    def __init__(self, name):
        """
        Arguments:
        name -- name of this job
        """
        super(Job, self).__init__()

        # create a unique ID
        self._id = '%s-%s' % (time.strftime('%Y%m%d-%H%M%S'), os.urandom(2).encode('hex'))
        self._dir = os.path.join(config_option('jobs_dir'), self._id)
        self._name = name
        self.pickver_job = PICKLE_VERSION
        self.tasks = []
        self.exception = None

        os.mkdir(self._dir)


    def __getstate__(self):
        """
        Used when saving a pickle file
        """
        d = self.__dict__.copy()
        # Isn't linked to state
        if '_dir' in d:
            del d['_dir']

        return d

    def __setstate__(self, state):
        """
        Used when loading a pickle file
        """
        self.__dict__ = state

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
            path = os.path.relpath(path, config_option('jobs_dir'))
        return str(path)

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

    def job_type(self):
        """
        String representation for this class
        virtual function
        """
        raise NotImplementedError('Implement me!')

    def on_status_update(self):
        """
        Called when StatusCls.status.setter is used
        """
        from digits.webapp import app, socketio

        message = {
                'update': 'status',
                'status': self.status.name,
                'css': self.status.css,
                'running': self.status.is_running(),
                }
        with app.app_context():
            message['html'] = render_template('status_updates.html', updates=self.status_history)

        socketio.emit('job update',
                message,
                namespace='/jobs',
                room=self.id(),
                )

    def abort(self):
        """
        Abort a job and stop all running tasks
        """
        if self.status.is_running():
            self.status = Status.ABORT
        for task in self.tasks:
            task.abort()

    def save(self):
        """save to pickle file"""
        try:
            # use tmpfile so we don't abort during pickle dump (leading to EOFErrors)
            tmpfile_path = self.path(self.SAVE_FILE + '.tmp')
            with open(tmpfile_path, 'wb') as tmpfile:
                pickle.dump(self, tmpfile)
            shutil.move(tmpfile_path, self.path(self.SAVE_FILE))
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print 'Caught %s while saving job: %s' % (type(e).__name__, e)

