# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .adapter import db
from .user import User
from .utils import WithRepr, WithAttributes, WithFiles, WithStatus


class Job(db.Model, WithRepr, WithAttributes, WithFiles, WithStatus):
    REPR_FIELDS = ['name', 'type', 'directory']

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('%s.id' % User.__tablename__))
    user = db.relationship(User.__name__, backref='jobs')
    name = db.Column(db.String(255), nullable=False)
    type = db.Column(db.String(50))
    notes = db.Column(db.Text)
    directory = db.Column(db.String(255), nullable=False, unique=True)
