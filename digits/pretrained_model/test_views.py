# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
import itertools
import json
import os
import re
import shutil
import tempfile
import time
import unittest
import urllib
import io
import tarfile

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
from digits.inference import GradientAscentJob

import digits.model.images.classification.test_views
import digits.model.views
import digits.pretrained_model.views
import digits.test_views
import digits.webapp

# Must import after importing digit.config
import caffe_pb2

# May be too short on a slow system
TIMEOUT_DATASET = 45
TIMEOUT_MODEL = 60

class BaseViewsTestWithPretrainedModel(digits.model.images.classification.test_views.BaseViewsTestWithModel):
    """
    Provides a model
    """
    @classmethod
    def setUpClass(cls):
        super(BaseViewsTestWithPretrainedModel, cls).setUpClass()
        job = digits.model.views.create_pretrained_model(cls.model_id,None,-1)
        cls.model_id = job.id()

class BaseTestInference(BaseViewsTestWithPretrainedModel):
    """
    Test Inference Tasks Such As Gradient Ascent
    """
    def test_max_activations(self):
        """ Run Gradient Ascent on a softmax layer for first unit """

        # Get pretrained model job:
        job = digits.webapp.scheduler.get_job(self.model_id)

        layer_name = "softmax"
        units = [0]

        gradient_ascent_job = digits.pretrained_model.views.create_max_activation_job(
            job,
            None,
            layer_name,
            units
        )

        gradient_ascent_job.wait_completion()
        activations = job.get_max_activations_path()

        assert os.path.isfile(activations), 'Gradient Ascent Job Failed'

class BaseTestUpload(digits.model.images.classification.test_views.BaseViewsTestWithModel):
    """
    Tests uploading Pretrained Models
    """
    def test_upload_manual(self):
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
                'image_type': info["image dimensions"][2],
                'resize_mode': info["image resize mode"],
                'width': info["image dimensions"][0],
                'height': info["image dimensions"][1],
                'job_name': 'test_create_pretrained_model_job'
                }
            )
        s = BeautifulSoup(rv.data, 'html.parser')
        body = s.select('body')

        assert rv.status_code == 302, 'POST failed with %s\n\n%s' % (rv.status_code, body)

    def test_upload_archive(self):
        job = digits.webapp.scheduler.get_job(self.model_id)

        if job is None:
            raise AssertionError('Failed To Create Job')

        info = json.dumps(job.json_dict(verbose=False,epoch=-1), sort_keys=True, indent=4, separators=(',', ': '))
        info_io = io.BytesIO()
        info_io.write(info)

        tmp = tempfile.NamedTemporaryFile()

        tf  = tarfile.open(fileobj=tmp, mode='w:')
        for path, name in job.download_files(-1):
            tf.add(path, arcname=name)

        tf_info = tarfile.TarInfo("info.json")
        tf_info.size = len(info_io.getvalue())
        info_io.seek(0)
        tf.addfile(tf_info, info_io)
        tmp.flush()
        tmp.seek(0)

        rv = self.app.post(
            '/pretrained_models/upload_archive',
            data = {
                'archive': tmp
                }
            )
        s = BeautifulSoup(rv.data, 'html.parser')
        body = s.select('body')
        tmp.close()

        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)

class TestCaffeInference(BaseTestInference):
    FRAMEWORK = 'caffe'

class TestCaffeUpload(BaseTestUpload):
    FRAMEWORK = 'caffe'

class TestTorchUpload(BaseTestUpload):
    FRAMEWORK = 'torch'
