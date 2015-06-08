# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import re
import os
import json
import shutil
import tempfile
import time
import unittest
import itertools
import urllib

import mock
import flask
from gevent import monkey
monkey.patch_all()
from bs4 import BeautifulSoup
import PIL.Image
from urlparse import urlparse
from cStringIO import StringIO

try:
    import caffe_pb2
except ImportError:
    # See issue #32
    from caffe.proto import caffe_pb2

import digits.webapp
import digits.test_views
import digits.model.images.classification.test_views
from digits.config import config_value

# May be too short on a slow system
TIMEOUT_DATASET = 15
TIMEOUT_MODEL = 20
TIMEOUT_EVALUATION = 60

################################################################################
# Base classes (they don't start with "Test" so nose won't run them)
################################################################################

class BaseViewsTest(digits.test_views.BaseViewsTest):
    """
    Provides some functions
    """ 
    @classmethod
    def evaluation_exists(cls, job_id):
        return cls.job_exists(job_id, 'evaluations')

    @classmethod
    def evaluation_status(cls, job_id):
        return cls.job_status(job_id, 'evaluations')

    @classmethod
    def abort_evaluation(cls, job_id):
        return cls.abort_job(job_id, job_type='evaluations')

    @classmethod
    def evaluation_wait_completion(cls, job_id, **kwargs):
        kwargs['job_type'] = 'evaluations'
        if 'timeout' not in kwargs:
            kwargs['timeout'] = TIMEOUT_EVALUATION
        return cls.job_wait_completion(job_id, **kwargs)

    @classmethod
    def delete_evaluation(cls, job_id):
        return cls.delete_job(job_id, job_type='evaluations')
 

class BaseViewsTestWithModel(BaseViewsTest,
        digits.model.images.classification.test_views.BaseViewsTestWithModel):
    """
    Provides a model 
    """
 
    @classmethod
    def setUpClass(cls):
        super(BaseViewsTestWithModel, cls).setUpClass()
        cls.created_evaluations = []

    @classmethod
    def tearDownClass(cls):
        # delete any created datasets
        for job_id in cls.created_evaluations:
            cls.delete_evaluation(job_id)
        super(BaseViewsTestWithModel, cls).tearDownClass()

    @classmethod
    def create_evaluation(cls, model_id=None, **kwargs):
        """
        Create an evaluation from a model
        Returns the job_id
        Raise RuntimeError if job fails to create

        Keyword arguments:
        **kwargs -- data to be sent with POST request
        """
        data = {
                'snapshot_epoch':   '1',
                'framework' :       cls.FRAMEWORK
                }
        data.update(kwargs)

        request_json = data.pop('json', False)
        url = '/evaluations/images/classification'
        if request_json:
            url += '.json'

        if model_id is None:
            model_id = cls.model_id
        url += '?job_id=' + model_id

        rv = cls.app.post(url, data=data)
        print url

        if request_json:
            if rv.status_code != 200:
                print json.loads(rv.data)
                raise RuntimeError('Evaluation creation failed with %s' % rv.status_code)
            return json.loads(rv.data)['id']

        # expect a redirect
        if not 300 <= rv.status_code <= 310:
            s = BeautifulSoup(rv.data)
            div = s.select('div.alert-danger')
            if div:
                raise RuntimeError(div[0])
            else:
                raise RuntimeError('Failed to create evaluation')

        job_id = cls.job_id_from_response(rv)
        assert cls.evaluation_exists(job_id), 'evaluation not found after successful creation'

        cls.created_evaluations.append(job_id)
        return job_id


class BaseViewsTestWithEvaluation(BaseViewsTestWithModel):
    """
    Provides an evaluation
    """
    @classmethod
    def setUpClass(cls):
        super(BaseViewsTestWithEvaluation, cls).setUpClass()
        assert cls.model_wait_completion(cls.model_id) == 'Done', 'model create failed'
        cls.evaluation_id = cls.create_evaluation(json=True)
        assert cls.evaluation_wait_completion(cls.evaluation_id) == 'Done', 'create failed'

class BaseTestViews(BaseViewsTest):
    """
    Tests which don't require a dataset or a model
    """

    def test_nonexistent_evaluation(self):
        assert not self.evaluation_exists('foo'), "evaluation shouldn't exist"



class BaseTestCreation(BaseViewsTestWithModel):
    """
    Model creation tests
    """
    def test_create_json(self):
        job_id = self.create_evaluation(json=True)
        self.abort_evaluation(job_id)

    def test_create_delete(self):
        job_id = self.create_evaluation()
        assert self.delete_evaluation(job_id) == 200, 'delete failed'
        assert not self.evaluation_exists(job_id), 'evaluation exists after delete'

    def test_create_wait_delete(self):
        job_id = self.create_evaluation()
        assert self.evaluation_wait_completion(job_id) == 'Done', 'create failed'
        assert self.delete_evaluation(job_id) == 200, 'delete failed'
        assert not self.evaluation_exists(job_id), 'evaluation exists after delete'

    def test_create_abort_delete(self):
        job_id = self.create_evaluation()
        assert self.abort_evaluation(job_id) == 200, 'abort failed'
        assert self.delete_evaluation(job_id) == 200, 'delete failed'
        assert not self.evaluation_exists(job_id), 'evaluation exists after delete'

    def test_evaluate_snapshot_2(self):
        model_id = self.create_model(train_epochs=2)
        assert self.model_wait_completion(model_id) == 'Done', 'model create failed'

        job_id = self.create_evaluation(model_id=model_id, snapshot_epoch=2)
        assert self.evaluation_wait_completion(job_id) == 'Done', 'evaluation create failed'

        rv = self.app.get('/evaluations/%s.json' % job_id)
        assert rv.status_code == 200, 'json load failed with %s' % rv.status_code 
  

class BaseTestCreated(BaseViewsTestWithEvaluation):
    """
    Tests on an evaluation that has already been created
    """
    def test_save(self):
        job = digits.webapp.scheduler.get_job(self.evaluation_id)
        assert job.save(), 'Job failed to save'
  
    def test_evaluation_json(self):
        rv = self.app.get('/evaluations/%s.json' % self.evaluation_id)
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        assert content['id'] == self.evaluation_id, 'id %s != %s' % (content['id'], self.evaluation_id)
        assert content['model_id'] == self.model_id, 'model_id %s != %s' % (content['model_id'], self.model_id)


class BaseTestModelEvaluationInteractions(BaseViewsTestWithModel):
    """
    Test the interactions between models and evaluations
    """
    # If you try to create an evaluation using a deleted model, it should fail
    def test_create_evaluation_deleted_model(self):
        model_id = self.create_model()
        assert self.delete_model(model_id) == 200, 'delete failed'
        assert not self.model_exists(model_id), 'model exists after delete'

        try:
            evaluation_id = self.create_evaluation(model_id=model_id)
        except RuntimeError:
            return
        assert False, 'Should have failed'

    # If you try to create an evaluation using a model epoch
    #   which has not been trained, it should fail
    def test_create_model_running_dataset(self):
        model_id = self.create_model(train_epochs=1)

        try:
            evaluation_id = self.create_evaluation(model_id=model_id, snapshot_epoch=2)
        except RuntimeError:
            return

        assert False, 'Should have failed'
 
    # If you try to delete a completed model with a dependent evaluation, 
    # it should delete the evaluation as well 
    def test_delete_model_dependent_evaluation(self):
        model_id = self.create_model()
        assert self.model_wait_completion(model_id) == 'Done', 'model creation failed'
        evaluation_id = self.create_evaluation(model_id=model_id)
        assert self.delete_model(model_id) == 200, 'model deletion should succeed'
        assert not self.evaluation_exists(evaluation_id), 'evaluation exists after delete'

    # If you try to delete a running model with a dependent evaluation, 
    # it should delete this dependent evaluation as well
    # def test_delete_running_model_dependent_evaluation(self):
        # dataset_id = self.create_dataset()
        # model_id = self.create_model(dataset=dataset_id)
        # assert self.delete_dataset(dataset_id) == 403, 'dataset deletion should not have succeeded'
        # self.abort_dataset(dataset_id)
        # self.abort_model(model_id)


# class BaseTestCreatedWide(BaseTestCreated):
#     IMAGE_WIDTH = 20

# class BaseTestCreatedTall(BaseTestCreated):
#     IMAGE_HEIGHT = 20

# class BaseTestCreatedCropInForm(BaseTestCreated):
#     CROP_SIZE = 8

# class BaseTestCreatedCropInNetwork(BaseTestCreated):
#     CAFFE_NETWORK = \
# """
# layer {
#   name: "data"
#   type: "Data"
#   top: "data"
#   top: "label"
#   include {
#     phase: TRAIN
#   }
#   transform_param {
#     crop_size: 8
#   }
# }
# layer {
#   name: "data"
#   type: "Data"
#   top: "data"
#   top: "label"
#   include {
#     phase: TEST
#   }
#   transform_param {
#     crop_size: 8
#   }
# }
# layer {
#     name: "hidden"
#     type: 'InnerProduct'
#     bottom: "data"
#     top: "output"
# }
# layer {
#     name: "loss"
#     type: "SoftmaxWithLoss"
#     bottom: "output"
#     bottom: "label"
#     top: "loss"
# }
# layer {
#     name: "accuracy"
#     type: "Accuracy"
#     bottom: "output"
#     bottom: "label"
#     top: "accuracy"
#     include {
#         phase: TEST
#     }
# }
# """

################################################################################
# Test classes
################################################################################

class TestCaffeViews(BaseTestViews):
    FRAMEWORK = 'caffe'

class TestCaffeCreation(BaseTestCreation):
    FRAMEWORK = 'caffe'

class TestCaffeCreated(BaseTestCreated):
    FRAMEWORK = 'caffe'

class TestCaffeModelEvaluationInteractions(BaseTestModelEvaluationInteractions):
    FRAMEWORK = 'caffe'

# class TestCaffeCreatedWide(BaseTestCreatedWide):
#     FRAMEWORK = 'caffe'

# class TestCaffeCreatedTall(BaseTestCreatedTall):
#     FRAMEWORK = 'caffe'

# class TestCaffeCreatedCropInForm(BaseTestCreatedCropInForm):
#     FRAMEWORK = 'caffe'

# class TestCaffeCreatedCropInNetwork(BaseTestCreatedCropInNetwork):
#     FRAMEWORK = 'caffe'

# class TestCaffeLeNet(TestCaffeCreated):
#     IMAGE_WIDTH = 28
#     IMAGE_HEIGHT = 28

#     CAFFE_NETWORK=open(
#             os.path.join(
#                 os.path.dirname(digits.__file__),
#                 'standard-networks', 'caffe', 'lenet.prototxt')
#             ).read()

