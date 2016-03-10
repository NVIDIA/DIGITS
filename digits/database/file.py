# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from .adapter import db
from .dataset import Dataset
from .inference import Inference
from .job import Job
from .model import Model
from .task import Task
from .training import Training
from .utils import WithRepr


class File(db.Model, WithRepr):
    REPR_FIELDS = ['key', 'path']

    id = db.Column(db.Integer, primary_key=True)
    type = db.Column(db.String(255), nullable=False)
    __mapper_args__ = {'polymorphic_on': type}
    key = db.Column(db.String(255), nullable=False)
    path = db.Column(db.String(255), nullable=False)


class DatasetFile(File):
    __mapper_args__ = {'polymorphic_identity': 'dataset'}

    dataset_id = db.Column(
        db.Integer,
        db.ForeignKey('%s.id' % Dataset.__tablename__),
    )
    dataset = db.relationship(
        Dataset.__name__,
        backref=db.backref('files', lazy='dynamic'),
    )


class InferenceFile(File):
    __mapper_args__ = {'polymorphic_identity': 'inference'}

    inference_id = db.Column(
        db.Integer,
        db.ForeignKey('%s.id' % Inference.__tablename__),
    )
    inference = db.relationship(
        Inference.__name__,
        backref=db.backref('files', lazy='dynamic'),
    )


class JobFile(File):
    __mapper_args__ = {'polymorphic_identity': 'job'}

    job_id = db.Column(
        db.Integer,
        db.ForeignKey('%s.id' % Job.__tablename__),
    )
    job = db.relationship(
        Job.__name__,
        backref=db.backref('files', lazy='dynamic'),
    )


class ModelFile(File):
    __mapper_args__ = {'polymorphic_identity': 'model'}

    model_id = db.Column(
        db.Integer,
        db.ForeignKey('%s.id' % Model.__tablename__),
    )
    model = db.relationship(
        Model.__name__,
        backref=db.backref('files', lazy='dynamic'),
    )


class TaskFile(File):
    __mapper_args__ = {'polymorphic_identity': 'task'}

    task_id = db.Column(
        db.Integer,
        db.ForeignKey('%s.id' % Task.__tablename__),
    )
    task = db.relationship(
        Task.__name__,
        backref=db.backref('files', lazy='dynamic'),
    )


class TrainingFile(File):
    __mapper_args__ = {'polymorphic_identity': 'training'}

    training_id = db.Column(
        db.Integer,
        db.ForeignKey('%s.id' % Training.__tablename__),
    )
    training = db.relationship(
        Training.__name__,
        backref=db.backref('files', lazy='dynamic'),
    )

