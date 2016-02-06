# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import datetime

from .adapter import db
from .job import Job
from .utils import WithRepr


class JobStatusUpdate(db.Model, WithRepr):
    REPR_FIELDS = ['status', 'timestamp']

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
