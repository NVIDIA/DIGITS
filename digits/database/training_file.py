# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .adapter import db, my_repr
from .training import Training

class TrainingFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    training_id = db.Column(db.Integer,
                            db.ForeignKey('%s.id' % Training.__tablename__),
                            nullable=False,
                            )
    training = db.relationship(Training.__name__,
                               backref=db.backref('files', lazy='dynamic'))
    label = db.Column(db.String(255))
    path = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return my_repr(self, ['path', 'label'])
