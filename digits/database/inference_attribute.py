# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .adapter import db, my_repr
from .inference import Inference

class InferenceAttribute(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    inference_id = db.Column(db.Integer,
                         db.ForeignKey('%s.id' % Inference.__tablename__),
                         nullable=False,
                         )
    inference = db.relationship(Inference.__name__,
                                backref=db.backref('attributes', lazy='dynamic'))
    key = db.Column(db.String(255), nullable=False)
    value = db.Column(db.String(255))

    def __repr__(self):
        return my_repr(self, ['key', 'value'])
