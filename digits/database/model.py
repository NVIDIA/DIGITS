# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .adapter import db, my_repr
from .task import Task
from .training import Training

class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.Integer, db.ForeignKey('%s.id' % Task.__tablename__))
    task = db.relationship(Task.__name__, backref='models')
    training_id = db.Column(db.Integer, db.ForeignKey('%s.id' % Training.__tablename__))
    training = db.relationship(Training.__name__, backref='models')

    def __repr__(self):
        return my_repr(self)
