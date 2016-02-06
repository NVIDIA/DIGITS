# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .adapter import db
from .task import Task
from .utils import WithRepr


class TaskAttribute(db.Model, WithRepr):
    REPR_FIELDS = ['key', 'value']

    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.Integer,
                        db.ForeignKey('%s.id' % Task.__tablename__),
                        nullable=False,
                        )
    task = db.relationship(Task.__name__,
                           backref=db.backref('attributes', lazy='dynamic'))
    key = db.Column(db.String(255), nullable=False)
    value = db.Column(db.String(255))
