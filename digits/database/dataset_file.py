# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .adapter import db, my_repr
from .dataset import Dataset

class DatasetFile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer,
                           db.ForeignKey('%s.id' % Dataset.__tablename__),
                           nullable=False,
                           )
    dataset = db.relationship(Dataset.__name__,
                              backref=db.backref('files', lazy='dynamic'))
    label = db.Column(db.String(255))
    path = db.Column(db.String(255), nullable=False)

    def __repr__(self):
        return my_repr(self, ['path', 'label'])
