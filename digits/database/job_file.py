# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .adapter import db
from .job import Job
from .utils import WithRepr


class JobFile(db.Model, WithRepr):
    REPR_FIELDS = ['key', 'path']

    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer,
                       db.ForeignKey('%s.id' % Job.__tablename__),
                       nullable=False,
                       )
    job = db.relationship(Job.__name__,
                          backref=db.backref('files', lazy='dynamic'))
    key = db.Column(db.String(255), nullable=False)
    path = db.Column(db.String(255), nullable=False)
