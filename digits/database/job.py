# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .adapter import db, my_repr
from .user import User

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('%s.id' % User.__tablename__))
    user = db.relationship(User.__name__, backref='jobs')
    name = db.Column(db.String(255), nullable=False)
    type = db.Column(db.String(50))
    notes = db.Column(db.Text)
    directory = db.Column(db.String(255), nullable=False, unique=True)

    def __repr__(self):
        return my_repr(self, ['name', 'type', 'directory'])
