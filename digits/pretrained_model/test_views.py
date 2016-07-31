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
from digits.pretrained_model import PretrainedModelJob
import digits.webapp
import digits.dataset.images.classification.test_views
import digits.model.images.classification.test_views
import digits.test_views

# Must import after importing digit.config
import caffe_pb2

# May be too short on a slow system
TIMEOUT_DATASET = 45
TIMEOUT_MODEL = 60

class BaseTestUpload(digits.model.images.classification.test_views.BaseViewsTestWithModel):
    """
    Tests uploading Pretrained Models
    """
    def test_upload(self):
        # job = digits.webapp.scheduler.get_job(self.model_id)
        job = digits.webapp.scheduler.get_job(self.model_id)

        if job is None:
            raise AssertionError('Failed To Create Job')

        # Write the stats of the job to json,
        # and store in tempfile (for archive)
        info = job.json_dict(verbose=False,epoch=-1)
        task = job.train_task()

        snapshot_filename = task.get_snapshot(-1)
        weights_file = open(snapshot_filename, 'r')
        model_def_file = open(os.path.join(job.dir(),task.model_file), 'r')
        labels_file = open(os.path.join(task.dataset.dir(),info["labels file"]), 'r')

        rv = self.app.post(
            '/pretrained_models/new',
            data = {
                'weights_file': weights_file,
                'model_def_file': model_def_file,
                'labels_file': labels_file,
                'framework': info['framework'],
                'job_name': info['name'],
                'image_type': info["image dimensions"][2],
                'resize_mode': info["image resize mode"],
                'width': info["image dimensions"][0],
                'height': info["image dimensions"][1]
                }
            )
        s = BeautifulSoup(rv.data, 'html.parser')
        body = s.select('body')

        assert rv.status_code == 302, 'POST failed with %s\n\n%s' % (rv.status_code, body)


class TestCaffeManualUpload(BaseTestUpload):
    FRAMEWORK = 'caffe'

class TestTorchManualUpload(BaseTestUpload):
    FRAMEWORK = 'torch'
