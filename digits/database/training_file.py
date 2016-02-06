# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .adapter import db
from .training import Training
from .utils import WithRepr


class TrainingFile(db.Model, WithRepr):
    REPR_FIELDS = ['key', 'path']

    id = db.Column(db.Integer, primary_key=True)
    training_id = db.Column(db.Integer,
                            db.ForeignKey('%s.id' % Training.__tablename__),
                            nullable=False,
                            )
    training = db.relationship(Training.__name__,
                               backref=db.backref('files', lazy='dynamic'))
    key = db.Column(db.String(255), nullable=False)
    path = db.Column(db.String(255), nullable=False)
