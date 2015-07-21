# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import re
import os
import tempfile

import unittest
import mock
import flask
from bs4 import BeautifulSoup
try:
    import caffe_pb2
except ImportError:
    # See issue #32
    from caffe.proto import caffe_pb2

from . import views as _
import digits
from digits.webapp import app, scheduler

class BaseTestCase(object):
    @classmethod
    def setupClass(cls):
        app.config['TESTING'] = True
        cls.app = app.test_client()
        cls.server = 'http://0.0.0.0:5000/'
        cls.jobs = []
        scheduler.running = True

    @classmethod
    def tearDownClass(cls):
        scheduler.jobs = []
        scheduler.running = False

class TestCreate(BaseTestCase):

    @staticmethod
    def get_error_msg(html):
        s = BeautifulSoup(html)
        div = s.select('div.alert-danger')
        if div:
            return str(div[0])
        else:
            return None

    @classmethod
    def setupClass(cls):
        super(TestCreate, cls).setupClass()

        with app.test_request_context():
            cls.url = flask.url_for('image_classification_model_create')

        dj = mock.Mock(spec=digits.dataset.ImageClassificationDatasetJob)
        dj.status.is_running.return_value = True
        dj.id.return_value = 'dataset'
        dj.name.return_value = ''

        mj = mock.Mock(spec=digits.model.ImageClassificationModelJob)
        mj.id.return_value = 'model'
        mj.name.return_value = ''
        _, cls.temp_snapshot_path = tempfile.mkstemp() #instead of using a dummy hardcoded value as snapshot path, temp file path is used to avoid the filen't exists exception in views.py.
        mj.train_task.return_value.snapshots = [(cls.temp_snapshot_path, 1)]
        mj.train_task.return_value.network = caffe_pb2.NetParameter()

        digits.webapp.scheduler.jobs = [dj, mj]

    @classmethod
    def tearDownClass(cls):
        super(TestCreate, cls).tearDownClass()
        try:
            os.remove(cls.temp_snapshot_path)
        except OSError:
            pass

    def test_empty_request(self):
        """empty request"""
        rv = self.app.post(self.url)
        assert rv.status_code == 400
        assert 'model_name' in self.get_error_msg(rv.data)

    def test_crop_size(self):
        """custom crop size"""

        rv = self.app.post(self.url, data={
            'method': 'standard',
            'dataset': 'dataset',
            'crop_size': 12,
            'standard_networks': 'lenet',
            'model_name': 'test',
            })

        if not (300 <= rv.status_code <= 310):
            msg = self.get_error_msg(rv.data)
            if msg is not None:
                raise RuntimeError(msg)
            else:
                raise RuntimeError('Failed to create model')

        assert scheduler.jobs[-1].train_task().crop_size == 12, \
                'crop size not saved properly'

    def test_previous_network_pretrained_model(self):
        """previous network, pretrained model"""

        rv = self.app.post(self.url, data={
            'method': 'previous',
            'model_name': 'test',
            'dataset': 'dataset',
            'previous_networks': 'model',
            'model-snapshot' : 1
            })

        if not (300 <= rv.status_code <= 310):
            msg = self.get_error_msg(rv.data)
            if msg is not None:
                raise RuntimeError(msg)
            else:
                raise RuntimeError('Failed to create model')

        assert scheduler.jobs[-1].train_task().pretrained_model == self.temp_snapshot_path, \
                'pretrained model not saved properly'

