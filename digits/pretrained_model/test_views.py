import itertools
import json
import os
import re
import shutil
import tempfile
import time
import unittest
import urllib

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

from bs4 import BeautifulSoup
import flask
import mock
import PIL.Image
from urlparse import urlparse

from digits.config import config_value
import digits.dataset.images.classification.test_views
import digits.test_views
import digits.webapp

# Must import after importing digit.config
import caffe_pb2

# May be too short on a slow system
TIMEOUT_DATASET = 45
TIMEOUT_MODEL = 60

################################################################################
# Base classes (they don't start with "Test" so nose won't run them)
################################################################################

class BaseViewsTest(digits.test_views.BaseViewsTest):
    """
    Provides some functions
    """
    CAFFE_NETWORK = \
"""
layer {
    name: "hidden"
    type: 'InnerProduct'
    bottom: "data"
    top: "output"
}
layer {
    name: "loss"
    type: "SoftmaxWithLoss"
    bottom: "output"
    bottom: "label"
    top: "loss"
    exclude { stage: "deploy" }
}
layer {
    name: "accuracy"
    type: "Accuracy"
    bottom: "output"
    bottom: "label"
    top: "accuracy"
    include { stage: "val" }
}
layer {
    name: "softmax"
    type: "Softmax"
    bottom: "output"
    top: "softmax"
    include { stage: "deploy" }
}
"""
    @classmethod
    def delete_pretrained_model(cls, job_id):
        return cls.delete_job(job_id, job_type='pretrained_models')

    @classmethod
    def network(cls):
        return cls.CAFFE_NETWORK

class TestViews(BaseViewsTest):

    def test_create_pretrained_caffe_model(self):
        """
        Test uploading a pretrained model for caffe
        """
        weights_file = (StringIO("..."), 'weights.caffemodel')
        model_def_file = (StringIO(self.network()), 'original.prototxt')
        labels_file = (StringIO("..."), 'labels.txt')

        rv = self.app.post(
            '/pretrained_models/new?job_id=%s',
            data = {
                'weights_file': weights_file,
                'model_def_file': model_def_file,
                'labels_file': labels_file,
                'framework': 'caffe',
                'job_name': 'test_create_pretrained_model_job'
                }
            )
        s = BeautifulSoup(rv.data, 'html.parser')
        body = s.select('body')

        assert rv.status_code == 302, 'POST failed with %s\n\n%s' % (rv.status_code, body)
