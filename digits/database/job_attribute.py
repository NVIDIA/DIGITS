# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .adapter import db, my_repr
from .job import Job

class JobAttribute(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer,
                       db.ForeignKey('%s.id' % Job.__tablename__),
                       nullable=False,
                       )
    job = db.relationship(Job.__name__,
                          backref=db.backref('attributes', lazy='dynamic'))
    key = db.Column(db.String(255), nullable=False)
    value = db.Column(db.String(255))

    def __repr__(self):
        return my_repr(self, ['key', 'value'])
