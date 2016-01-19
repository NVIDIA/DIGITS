# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import datetime

from .adapter import db, my_repr
from .job import Job

class JobStatusUpdate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer,
                       db.ForeignKey('%s.id' % Job.__tablename__),
                       nullable=False,
                       )
    job = db.relationship(Job.__name__,
                          backref=db.backref('status_updates',
                                             lazy='dynamic')
                          )
    status = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime,
                          default=datetime.datetime.utcnow,
                          )

    def __repr__(self):
        return my_repr(self, ['status', 'timestamp'])
