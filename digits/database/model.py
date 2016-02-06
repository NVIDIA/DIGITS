# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .adapter import db
from .task import Task
from .training import Training
from .utils import WithRepr, WithAttributes, WithFiles


class Model(db.Model, WithRepr, WithAttributes, WithFiles):
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.Integer, db.ForeignKey('%s.id' % Task.__tablename__))
    task = db.relationship(Task.__name__, backref='models')
    training_id = db.Column(db.Integer, db.ForeignKey('%s.id' % Training.__tablename__))
    training = db.relationship(Training.__name__, backref='models')
