# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .adapter import db
from .task import Task
from .dataset import Dataset
from .utils import WithRepr, WithAttributes, WithFiles


class Training(db.Model, WithRepr, WithAttributes, WithFiles):
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.Integer, db.ForeignKey('%s.id' % Task.__tablename__))
    task = db.relationship(Task.__name__, backref='trainings')


# Many-to-Many relationship
training_to_dataset = db.Table(
    'training_to_dataset',
    db.Column('training_id', db.Integer, db.ForeignKey('%s.id' % Training.__tablename__), primary_key=True),
    db.Column('dataset_id', db.Integer, db.ForeignKey('%s.id' % Dataset.__tablename__), primary_key=True),
)

Training.datasets = db.relationship(
    Dataset.__name__,
    secondary=training_to_dataset,
    backref=db.backref('trainings', lazy='dynamic'),
    lazy='dynamic',
)
