# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os
import time
import os.path
import pickle
import shutil

import flask

from digits import utils
from digits.config import config_value
from digits.utils import sizeof_fmt, filesystem as fs
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

    def __init__(self, name):
        """
        Arguments:
        name -- name of this job
        """
        super(Job, self).__init__()

        # create a unique ID
        self._id = '%s-%s' % (time.strftime('%Y%m%d-%H%M%S'), os.urandom(2).encode('hex'))
        self._dir = os.path.join(config_value('jobs_dir'), self._id)
        self._name = name
        self.pickver_job = PICKLE_VERSION
        self.tasks = []
        self.exception = None
        self._notes = None

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
        return str(path).replace("\\","/")

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
            print 'Caught %s while saving job: %s' % (type(e).__name__, e)
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
                          'percentage': int(round(100*progress)),
                      },
                      namespace='/jobs',
                      room='job_management'
                  )
