# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .adapter import db, my_repr
from .job import Job

class Task(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_id = db.Column(db.Integer,
                       db.ForeignKey('%s.id' % Job.__tablename__),
                       nullable=False,
                       )
    job = db.relationship(Job.__name__, backref='tasks')
    type = db.Column(db.String(50))
    progress = db.Column(db.Float, nullable=False, default=0, server_default='0')

    def __repr__(self):
        return my_repr(self, ['type'])


# Self-referencing Many-to-Many relationship
parents = db.Table(
    'task_parent',
    db.Column('parent_id', db.Integer, db.ForeignKey('%s.id' % Task.__tablename__), primary_key=True),
    db.Column('child_id', db.Integer, db.ForeignKey('%s.id' % Task.__tablename__), primary_key=True),
)

Task.parents = db.relationship(
    Task.__name__,
    secondary=parents,
    primaryjoin=(parents.c.child_id == Task.id),
    secondaryjoin=(parents.c.parent_id == Task.id),
    backref=db.backref('children', lazy='dynamic'),
    lazy='dynamic',
)
