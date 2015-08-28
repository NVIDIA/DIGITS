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
import digits.dataset.images.classification.test_views
from digits.config import config_value

# May be too short on a slow system
TIMEOUT_DATASET = 15
TIMEOUT_MODEL = 20

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
}
layer {
    name: "accuracy"
    type: "Accuracy"
    bottom: "output"
    bottom: "label"
    top: "accuracy"
    include {
        phase: TEST
    }
}
"""

    TORCH_NETWORK = \
"""
require 'nn'
require 'cunn'
local model = nn.Sequential()
model:add(nn.View(-1):setNumInputDims(3)) -- 10*10*3 -> 300
model:add(nn.Linear(300, 3))
model:add(nn.LogSoftMax())
model:cuda()
return model
"""

    @classmethod
    def setUpClass(cls):
        super(BaseViewsTest, cls).setUpClass()
        if cls.FRAMEWORK=='torch' and not config_value('torch_root'):
            raise unittest.SkipTest('Torch not found')

    @classmethod
    def model_exists(cls, job_id):
        return cls.job_exists(job_id, 'models')

    @classmethod
    def model_status(cls, job_id):
        return cls.job_status(job_id, 'models')

    @classmethod
    def abort_model(cls, job_id):
        return cls.abort_job(job_id, job_type='models')

    @classmethod
    def model_wait_completion(cls, job_id, **kwargs):
        kwargs['job_type'] = 'models'
        if 'timeout' not in kwargs:
            kwargs['timeout'] = TIMEOUT_MODEL
        return cls.job_wait_completion(job_id, **kwargs)

    @classmethod
    def delete_model(cls, job_id):
        return cls.delete_job(job_id, job_type='models')

    @classmethod
    def network(cls):
        return cls.TORCH_NETWORK if cls.FRAMEWORK=='torch' else cls.CAFFE_NETWORK

class BaseViewsTestWithDataset(BaseViewsTest,
        digits.dataset.images.classification.test_views.BaseViewsTestWithDataset):
    """
    Provides a dataset
    """

    # Inherited classes may want to override these attributes
    CROP_SIZE = None

    @classmethod
    def setUpClass(cls):
        super(BaseViewsTestWithDataset, cls).setUpClass()
        cls.created_models = []

    @classmethod
    def tearDownClass(cls):
        # delete any created datasets
        for job_id in cls.created_models:
            cls.delete_model(job_id)
        super(BaseViewsTestWithDataset, cls).tearDownClass()

    @classmethod
    def create_model(cls, **kwargs):
        """
        Create a model
        Returns the job_id
        Raise RuntimeError if job fails to create

        Keyword arguments:
        **kwargs -- data to be sent with POST request
        """
        data = {
                'model_name':                        'test_model',
                'dataset':                           cls.dataset_id,
                'method':                            'custom',
                'custom_network':                    cls.network(),
                'batch_size':                        10,
                'train_epochs':                      1,
                'framework' :                        cls.FRAMEWORK
                }
        if cls.CROP_SIZE is not None:
            data['crop_size'] = cls.CROP_SIZE
        data.update(kwargs)

        request_json = data.pop('json', False)
        url = '/models/images/classification'
        if request_json:
            url += '.json'

        rv = cls.app.post(url, data=data)

        if request_json:
            if rv.status_code != 200:
                print json.loads(rv.data)
                raise RuntimeError('Model creation failed with %s' % rv.status_code)
            return json.loads(rv.data)['id']

        # expect a redirect
        if not 300 <= rv.status_code <= 310:
            s = BeautifulSoup(rv.data)
            div = s.select('div.alert-danger')
            if div:
                raise RuntimeError(div[0])
            else:
                raise RuntimeError('Failed to create model')

        job_id = cls.job_id_from_response(rv)
        assert cls.model_exists(job_id), 'model not found after successful creation'

        cls.created_models.append(job_id)
        return job_id


class BaseViewsTestWithModel(BaseViewsTestWithDataset):
    """
    Provides a model
    """
    @classmethod
    def setUpClass(cls):
        super(BaseViewsTestWithModel, cls).setUpClass()
        cls.model_id = cls.create_model(json=True)
        assert cls.model_wait_completion(cls.model_id) == 'Done', 'create failed'

class BaseTestViews(BaseViewsTest):
    """
    Tests which don't require a dataset or a model
    """
    def test_page_model_new(self):
        rv = self.app.get('/models/images/classification/new')
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        assert 'New Image Classification Model' in rv.data, 'unexpected page format'

    def test_nonexistent_model(self):
        assert not self.model_exists('foo'), "model shouldn't exist"

    def test_visualize_network(self):
        if self.FRAMEWORK=='torch':
            raise unittest.SkipTest('Torch visualization not supported')
        rv = self.app.post('/models/visualize-network?framework='+self.FRAMEWORK,
                data = {'custom_network': self.network()}
                )
        s = BeautifulSoup(rv.data)
        body = s.select('body')
        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)
        image = s.select('img')
        assert image is not None, "didn't return an image"


class BaseTestCreation(BaseViewsTestWithDataset):
    """
    Model creation tests
    """
    def test_create_json(self):
        job_id = self.create_model(json=True)
        self.abort_model(job_id)

    def test_create_delete(self):
        job_id = self.create_model()
        assert self.delete_model(job_id) == 200, 'delete failed'
        assert not self.model_exists(job_id), 'model exists after delete'

    def test_create_wait_delete(self):
        job_id = self.create_model()
        assert self.model_wait_completion(job_id) == 'Done', 'create failed'
        assert self.delete_model(job_id) == 200, 'delete failed'
        assert not self.model_exists(job_id), 'model exists after delete'

    def test_create_abort_delete(self):
        job_id = self.create_model()
        assert self.abort_model(job_id) == 200, 'abort failed'
        assert self.delete_model(job_id) == 200, 'delete failed'
        assert not self.model_exists(job_id), 'model exists after delete'

    def test_snapshot_interval_2(self):
        job_id = self.create_model(snapshot_interval=0.5)
        assert self.model_wait_completion(job_id) == 'Done', 'create failed'
        rv = self.app.get('/models/%s.json' % job_id)
        assert rv.status_code == 200, 'json load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        assert len(content['snapshots']) > 1, 'should take >1 snapshot'

    def test_snapshot_interval_0_5(self):
        job_id = self.create_model(train_epochs=4, snapshot_interval=2)
        assert self.model_wait_completion(job_id) == 'Done', 'create failed'
        rv = self.app.get('/models/%s.json' % job_id)
        assert rv.status_code == 200, 'json load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        assert len(content['snapshots']) == 2, 'should take 2 snapshots'

    @unittest.skipIf(
            not config_value('gpu_list'),
            'no GPUs selected')
    @unittest.skipIf(
            not config_value('caffe_root')['cuda_enabled'],
            'CUDA disabled')
    @unittest.skipIf(
            config_value('caffe_root')['multi_gpu'],
            'multi-GPU enabled')
    def test_select_gpu(self):
        for index in config_value('gpu_list').split(','):
            yield self.check_select_gpu, index

    def check_select_gpu(self, gpu_index):
        job_id = self.create_model(select_gpu=gpu_index)
        assert self.model_wait_completion(job_id) == 'Done', 'create failed'

    @unittest.skipIf(
            not config_value('gpu_list'),
            'no GPUs selected')
    @unittest.skipIf(
            not config_value('caffe_root')['cuda_enabled'],
            'CUDA disabled')
    @unittest.skipIf(
            not config_value('caffe_root')['multi_gpu'],
            'multi-GPU disabled')
    def test_select_gpus(self):
        # test all possible combinations
        gpu_list = config_value('gpu_list').split(',')
        for i in xrange(len(gpu_list)):
            for combination in itertools.combinations(gpu_list, i+1):
                yield self.check_select_gpus, combination

    def check_select_gpus(self, gpu_list):
        job_id = self.create_model(select_gpus_list=','.join(gpu_list), batch_size=len(gpu_list))
        assert self.model_wait_completion(job_id) == 'Done', 'create failed'

    def test_retrain(self):
        job1_id = self.create_model()
        assert self.model_wait_completion(job1_id) == 'Done', 'first job failed'
        rv = self.app.get('/models/%s.json' % job1_id)
        assert rv.status_code == 200, 'json load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        assert len(content['snapshots']), 'should have at least snapshot'

        options = {
                'method': 'previous',
                'previous_networks': job1_id,
                }
        options['%s-snapshot' % job1_id] = content['snapshots'][-1]

        job_id = self.create_model(**options)
        assert self.model_wait_completion(job1_id) == 'Done', 'second job failed'


class BaseTestCreated(BaseViewsTestWithModel):
    """
    Tests on a model that has already been created
    """
    def test_save(self):
        job = digits.webapp.scheduler.get_job(self.model_id)
        assert job.save(), 'Job failed to save'

    def test_download(self):
        for extension in ['tar', 'zip', 'tar.gz', 'tar.bz2']:
            yield self.check_download, extension

    def check_download(self, extension):
        url = '/models/%s/download.%s' % (self.model_id, extension)
        rv = self.app.get(url)
        assert rv.status_code == 200, 'download "%s" failed with %s' % (url, rv.status_code)

    def test_index_json(self):
        rv = self.app.get('/index.json')
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        found = False
        for m in content['models']:
            if m['id'] == self.model_id:
                found = True
                break
        assert found, 'model not found in list'

    def test_model_json(self):
        rv = self.app.get('/models/%s.json' % self.model_id)
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        assert content['id'] == self.model_id, 'id %s != %s' % (content['id'], self.model_id)
        assert content['dataset_id'] == self.dataset_id, 'dataset_id %s != %s' % (content['dataset_id'], self.dataset_id)
        assert len(content['snapshots']) > 0, 'no snapshots in list'

    def test_classify_one(self):
        category = self.imageset_paths.keys()[0]
        image_path = self.imageset_paths[category][0]
        image_path = os.path.join(self.imageset_folder, image_path)
        with open(image_path,'rb') as infile:
            # StringIO wrapping is needed to simulate POST file upload.
            image_upload = (StringIO(infile.read()), 'image.png')

        rv = self.app.post(
                '/models/images/classification/classify_one?job_id=%s' % self.model_id,
                data = {
                    'image_file': image_upload,
                    'show_visualizations': 'y',
                    }
                )
        s = BeautifulSoup(rv.data)
        body = s.select('body')
        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)
        # gets an array of arrays [[confidence, label],...]
        predictions = [p.get_text().split() for p in s.select('ul.list-group li')]
        assert predictions[0][1] == category, 'image misclassified'

    def test_classify_one_json(self):
        category = self.imageset_paths.keys()[0]
        image_path = self.imageset_paths[category][0]
        image_path = os.path.join(self.imageset_folder, image_path)
        with open(image_path,'rb') as infile:
            # StringIO wrapping is needed to simulate POST file upload.
            image_upload = (StringIO(infile.read()), 'image.png')

        rv = self.app.post(
                '/models/images/classification/classify_one.json?job_id=%s' % self.model_id,
                data = {
                    'image_file': image_upload,
                    'show_visualizations': 'y',
                    }
                )
        assert rv.status_code == 200, 'POST failed with %s' % rv.status_code
        data = json.loads(rv.data)
        assert data['predictions'][0][0] == category, 'image misclassified'

    def test_classify_many(self):
        textfile_images = ''
        label_id = 0
        for label, images in self.imageset_paths.iteritems():
            for image in images:
                image_path = image
                image_path = os.path.join(self.imageset_folder, image_path)
                textfile_images += '%s %d\n' % (image_path, label_id)
            label_id += 1

        # StringIO wrapping is needed to simulate POST file upload.
        file_upload = (StringIO(textfile_images), 'images.txt')

        rv = self.app.post(
                '/models/images/classification/classify_many?job_id=%s' % self.model_id,
                data = {'image_list': file_upload}
                )
        s = BeautifulSoup(rv.data)
        body = s.select('body')
        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)

    def test_classify_many_json(self):
        textfile_images = ''
        label_id = 0
        for label, images in self.imageset_paths.iteritems():
            for image in images:
                image_path = image
                image_path = os.path.join(self.imageset_folder, image_path)
                textfile_images += '%s %d\n' % (image_path, label_id)
            label_id += 1

        # StringIO wrapping is needed to simulate POST file upload.
        file_upload = (StringIO(textfile_images), 'images.txt')

        rv = self.app.post(
                '/models/images/classification/classify_many.json?job_id=%s' % self.model_id,
                data = {'image_list': file_upload}
                )
        assert rv.status_code == 200, 'POST failed with %s' % rv.status_code
        data = json.loads(rv.data)
        assert 'classifications' in data, 'invalid response'

    def test_top_n(self):
        if self.FRAMEWORK=='torch':
            raise unittest.SkipTest('Torch top_n not supported')
        textfile_images = ''
        label_id = 0
        for label, images in self.imageset_paths.iteritems():
            for image in images:
                image_path = image
                image_path = os.path.join(self.imageset_folder, image_path)
                textfile_images += '%s %d\n' % (image_path, label_id)
            label_id += 1

        # StringIO wrapping is needed to simulate POST file upload.
        file_upload = (StringIO(textfile_images), 'images.txt')

        rv = self.app.post(
                '/models/images/classification/top_n?job_id=%s' % self.model_id,
                data = {'image_list': file_upload}
                )
        s = BeautifulSoup(rv.data)
        body = s.select('body')
        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)
        keys = self.imageset_paths.keys()
        for key in keys:
            assert key in rv.data, '"%s" not found in the response'

class BaseTestDatasetModelInteractions(BaseViewsTestWithDataset):
    """
    Test the interactions between datasets and models
    """
    # If you try to create a model using a deleted dataset, it should fail
    def test_create_model_deleted_dataset(self):
        dataset_id = self.create_dataset()
        assert self.delete_dataset(dataset_id) == 200, 'delete failed'
        assert not self.dataset_exists(dataset_id), 'dataset exists after delete'

        try:
            model_id = self.create_model(dataset=dataset_id)
        except RuntimeError:
            return
        assert False, 'Should have failed'

    # If you try to create a model using a running dataset,
    #   it should wait to start until the dataset is completed
    def test_create_model_running_dataset(self):
        dataset_id = self.create_dataset()
        model_id = self.create_model(dataset=dataset_id)

        # Model should be in WAIT status while dataset is running
        #   Copying functionality from job_wait_completion ...
        start_time = time.time()
        timeout = TIMEOUT_DATASET

        dataset_status = self.dataset_status(dataset_id)
        while dataset_status != 'Done':
            model_status = self.model_status(model_id)
            if model_status == 'Initialized':
                # give it some time ...
                pass
            elif model_status == 'Waiting':
                # That's what we were waiting for
                break
            else:
                raise Exception('Model not waiting - "%s"' % model_status)
            assert (time.time() - start_time) < timeout, 'Job took more than %s seconds' % timeout
            time.sleep(0.5)
            dataset_status = self.dataset_status(dataset_id)

        # Model should switch to RUN status after dataset is DONE
        assert self.dataset_wait_completion(dataset_id) == 'Done', 'dataset creation failed'
        time.sleep(1)
        assert self.model_status(model_id) in ['Running', 'Done'], "model didn't start"
        self.abort_model(model_id)

    # If you try to delete a completed dataset with a dependent model, it should fail
    def test_delete_dataset_dependent_model(self):
        dataset_id = self.create_dataset()
        model_id = self.create_model(dataset=dataset_id)
        assert self.dataset_wait_completion(dataset_id) == 'Done', 'dataset creation failed'
        assert self.delete_dataset(dataset_id) == 403, 'dataset deletion should not have succeeded'
        self.abort_model(model_id)

    # If you try to delete a running dataset with a dependent model, it should fail
    def test_delete_running_dataset_dependent_model(self):
        dataset_id = self.create_dataset()
        model_id = self.create_model(dataset=dataset_id)
        assert self.delete_dataset(dataset_id) == 403, 'dataset deletion should not have succeeded'
        self.abort_dataset(dataset_id)
        self.abort_model(model_id)


class BaseTestCreatedWide(BaseTestCreated):
    IMAGE_WIDTH = 20

class BaseTestCreatedTall(BaseTestCreated):
    IMAGE_HEIGHT = 20

class BaseTestCreatedCropInForm(BaseTestCreated):
    CROP_SIZE = 8

class BaseTestCreatedCropInNetwork(BaseTestCreated):
    CAFFE_NETWORK = \
"""
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    crop_size: 8
  }
}
layer {
  name: "data"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    crop_size: 8
  }
}
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
}
layer {
    name: "accuracy"
    type: "Accuracy"
    bottom: "output"
    bottom: "label"
    top: "accuracy"
    include {
        phase: TEST
    }
}
"""

################################################################################
# Test classes
################################################################################

class TestCaffeViews(BaseTestViews):
    FRAMEWORK = 'caffe'

class TestCaffeCreation(BaseTestCreation):
    FRAMEWORK = 'caffe'

class TestCaffeCreated(BaseTestCreated):
    FRAMEWORK = 'caffe'

class TestCaffeDatasetModelInteractions(BaseTestDatasetModelInteractions):
    FRAMEWORK = 'caffe'

class TestCaffeCreatedWide(BaseTestCreatedWide):
    FRAMEWORK = 'caffe'

class TestCaffeCreatedTall(BaseTestCreatedTall):
    FRAMEWORK = 'caffe'

class TestCaffeCreatedCropInForm(BaseTestCreatedCropInForm):
    FRAMEWORK = 'caffe'

class TestCaffeCreatedCropInNetwork(BaseTestCreatedCropInNetwork):
    FRAMEWORK = 'caffe'

class TestTorchViews(BaseTestViews):
    FRAMEWORK = 'torch'

class TestTorchCreation(BaseTestCreation):
    FRAMEWORK = 'torch'

class TestTorchCreated(BaseTestCreated):
    FRAMEWORK = 'torch'

class TestTorchDatasetModelInteractions(BaseTestDatasetModelInteractions):
    FRAMEWORK = 'torch'

class TestCaffeLeNet(TestCaffeCreated):
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28

    CAFFE_NETWORK=open(
            os.path.join(
                os.path.dirname(digits.__file__),
                'standard-networks', 'caffe', 'lenet.prototxt')
            ).read()





