# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import datetime

from .adapter import db
from .task import Task
from .utils import WithRepr


class TaskStatusUpdate(db.Model, WithRepr):
    REPR_FIELDS = ['status', 'timestamp']

    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.Integer,
                        db.ForeignKey('%s.id' % Task.__tablename__),
                        nullable=False,
                        )
    task = db.relationship(Task.__name__, backref='status_updates')
    status = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime,
                          default=datetime.datetime.utcnow,
                          )
