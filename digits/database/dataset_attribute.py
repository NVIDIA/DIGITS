# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .adapter import db
from .dataset import Dataset
from .utils import WithRepr

class DatasetAttribute(db.Model, WithRepr):
    REPR_FIELDS = ['key', 'value']

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer,
                           db.ForeignKey('%s.id' % Dataset.__tablename__),
                           nullable=False,
                           )
    dataset = db.relationship(Dataset.__name__,
                              backref=db.backref('attributes', lazy='dynamic'))
    key = db.Column(db.String(255), nullable=False)
    value = db.Column(db.String(255))
