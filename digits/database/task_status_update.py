# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import datetime

from .adapter import db, my_repr
from .task import Task

class TaskStatusUpdate(db.Model):
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

    def __repr__(self):
        return my_repr(self, ['status', 'timestamp'])
