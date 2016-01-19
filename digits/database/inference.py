# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .adapter import db, my_repr
from .model import Model
from .task import Task

class Inference(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.Integer, db.ForeignKey('%s.id' % Task.__tablename__))
    task = db.relationship(Task.__name__, backref='inferences')
    model_id = db.Column(db.Integer, db.ForeignKey('%s.id' % Model.__tablename__))
    model = db.relationship(Model.__name__, backref='inferences')

    def __repr__(self):
        return my_repr(self)
