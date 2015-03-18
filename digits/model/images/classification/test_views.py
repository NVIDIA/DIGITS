# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import re

import unittest
import mock
from flask import url_for
from bs4 import BeautifulSoup
from caffe.proto import caffe_pb2

from . import views as _
import digits

class BaseTestCase(object):
    @classmethod
    def setupClass(cls):
        digits.webapp.app.config['TESTING'] = True
        cls.app = digits.webapp.app.test_client()
        cls.server = 'http://0.0.0.0:5000/'

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

        with digits.webapp.app.test_request_context():
            cls.url = url_for('image_classification_model_create')

    def test_empty_request(self):
        """empty request"""
        rv = self.app.post(self.url)
        assert rv.status_code == 400
        assert 'model_name' in self.get_error_msg(rv.data)

    @unittest.skip('expected failure')
    def test_previous_network_pretrained_model(self):
        """previous network, pretrained model"""
        mock_dataset_job = mock.Mock(spec=digits.dataset.ImageClassificationDatasetJob)
        mock_dataset_job.status.is_running.return_value = True
        mock_dataset_job.id.return_value = 'dataset'
        mock_dataset_job.name.return_value = ''

        mock_model_job = mock.Mock(spec=digits.model.ImageClassificationModelJob)
        mock_model_job.id.return_value = 'model'
        mock_model_job.name.return_value = ''
        mock_model_job.train_task.return_value.snapshots = [('path', 1)]
        mock_model_job.train_task.return_value.network = caffe_pb2.NetParameter()
        digits.webapp.scheduler.running = True
        digits.webapp.scheduler.jobs = [mock_dataset_job, mock_model_job]

        rv = self.app.post(self.url, data={
            'method': 'previous',
            'model_name': 'test',
            'dataset': 'dataset',
            'previous_networks': 'model',
            # TODO: select snapshot 1
            })

        digits.webapp.scheduler.running = False
        jobs = digits.webapp.scheduler.jobs
        digits.webapp.scheduler.jobs = []

        assert jobs[-1].train_task().pretrained_model == 'path'
        assert False

