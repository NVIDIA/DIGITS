# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .adapter import db
from .model import Model
from .utils import WithRepr


class ModelAttribute(db.Model, WithRepr):
    REPR_FIELDS = ['key', 'value']

    id = db.Column(db.Integer, primary_key=True)
    model_id = db.Column(db.Integer,
                         db.ForeignKey('%s.id' % Model.__tablename__),
                         nullable=False,
                         )
    model = db.relationship(Model.__name__,
                            backref=db.backref('attributes', lazy='dynamic'))
    key = db.Column(db.String(255), nullable=False)
    value = db.Column(db.String(255))
