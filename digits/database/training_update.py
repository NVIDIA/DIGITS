# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .adapter import db
from .training import Training
from .utils import WithRepr


class TrainingUpdate(db.Model, WithRepr):
    REPR_FIELDS = ['epoch', 'iteration', 'name', 'value']

    id = db.Column(db.Integer, primary_key=True)
    training_id = db.Column(db.Integer,
                            db.ForeignKey('%s.id' % Training.__tablename__),
                            nullable=False,
                            )
    training = db.relationship(Training.__name__, backref='updates')
    epoch = db.Column(db.Float, nullable=False)
    phase = db.Column(db.String(255))
    name = db.Column(db.String(255), nullable=False)
    value = db.Column(db.Float)
