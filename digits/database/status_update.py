# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import datetime

from .adapter import db
from .job import Job
from .task import Task
from .utils import WithRepr


class StatusUpdate(db.Model, WithRepr):
    REPR_FIELDS = ['status', 'timestamp']

    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(255), nullable=False)
    __mapper_args__ = {'polymorphic_on': type}
    status = db.Column(db.String(1), nullable=False)
    timestamp = db.Column(
        db.DateTime,
        default=datetime.datetime.utcnow,
    )

    # Enum-ish
    INIT = 'I'
    WAIT =  'W'
    RUN =   'R'
    DONE =  'D'
    ABORT = 'A'
    ERROR = 'E'


    @property
    def name(self):
        """
        Return a human-readable representation of self.status
        """
        if self.status == self.INIT:
            return 'Initialized'
        elif self.status == self.WAIT:
            return 'Waiting'
        elif self.status == self.RUN:
            return 'Running'
        elif self.status == self.DONE:
            return 'Done'
        elif self.status == self.ABORT:
            return 'Aborted'
        elif self.status == self.ERROR:
            return 'Error'
        else:
            return '?'

    # Operators

    def __eq__(self, other):
        if type(other) == type(self):
            return self.status == other.status
        elif type(other) == str:
            return self.status == other
        else:
            return False

    def __ne__(self, other):
        if type(other) == type(self):
            return self.status != other.status
        elif type(other) == str:
            return self.status != other
        else:
            return True

    # Utility functions

    @property
    def name(self):
        """
        Return the CSS (color) associated with this status
        """
        if self.status == self.INIT:
            return 'warning'
        elif self.status == self.WAIT:
            return 'arning'
        elif self.status == self.RUN:
            return 'info'
        elif self.status == self.DONE:
            return 'success'
        elif self.status == self.ABORT:
            return 'warning'
        elif self.status == self.ERROR:
            return 'danger'
        else:
            return 'default'

    def is_running(self):
        return self.status in (self.INIT, self.WAIT, self.RUN)


class JobStatusUpdate(StatusUpdate):
    __mapper_args__ = {'polymorphic_identity': 'job'}

    job_id = db.Column(
        db.Integer,
        db.ForeignKey('%s.id' % Job.__tablename__),
    )
    job = db.relationship(
        Job.__name__,
        backref=db.backref('status_updates', lazy='dynamic'),
    )


class TaskStatusUpdate(StatusUpdate):
    __mapper_args__ = {'polymorphic_identity': 'task'}

    task_id = db.Column(
        db.Integer,
        db.ForeignKey('%s.id' % Task.__tablename__),
    )
    task = db.relationship(
        Task.__name__,
        backref=db.backref('status_updates', lazy='dynamic'),
    )
