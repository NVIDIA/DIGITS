# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

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
import caffe_pb2

import digits.webapp
import digits.test_views
import digits.dataset.images.generic.test_views
from digits.config import config_value

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
  name: "scale"
  type: "Power"
  bottom: "data"
  top: "scale"
  power_param {
    scale: 0.004
  }
}
layer {
  name: "hidden"
  type: "InnerProduct"
  bottom: "scale"
  top: "output"
  inner_product_param {
    num_output: 2
  }
}
layer {
  name: "train_loss"
  type: "EuclideanLoss"
  bottom: "output"
  bottom: "label"
  top: "loss"
}
"""

    TORCH_NETWORK = \
"""
return function(p)
    local nDim = 1
    if p.inputShape then p.inputShape:apply(function(x) nDim=nDim*x end) end
    local net = nn.Sequential()
    net:add(nn.MulConstant(0.004))
    net:add(nn.View(-1):setNumInputDims(3))  -- flatten
    net:add(nn.Linear(nDim,2)) -- c*h*w -> 2
    return {
        model = net,
        loss = nn.MSECriterion(),
    }
end
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
        digits.dataset.images.generic.test_views.BaseViewsTestWithDataset):
    """
    Provides a dataset
    """

    # Inherited classes may want to override these attributes
    CROP_SIZE = None
    TRAIN_EPOCHS = 3
    LR_POLICY = None
    LEARNING_RATE = None

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
    def create_model(cls, learning_rate=None, **kwargs):
        """
        Create a model
        Returns the job_id
        Raise RuntimeError if job fails to create

        Keyword arguments:
        **kwargs -- data to be sent with POST request
        """
        if learning_rate is None:
            learning_rate = cls.LEARNING_RATE
        data = {
                'model_name':       'test_model',
                'dataset':          cls.dataset_id,
                'method':           'custom',
                'custom_network':   cls.network(),
                'batch_size':       10,
                'train_epochs':     cls.TRAIN_EPOCHS,
                'random_seed':      0xCAFEBABE,
                'framework':        cls.FRAMEWORK,
                }
        if cls.CROP_SIZE is not None:
            data['crop_size'] = cls.CROP_SIZE
        if cls.LR_POLICY is not None:
            data['lr_policy'] = cls.LR_POLICY
        if learning_rate is not None:
            data['learning_rate'] = learning_rate
        data.update(kwargs)

        request_json = data.pop('json', False)
        url = '/models/images/generic'
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
            print 'Status code:', rv.status_code
            s = BeautifulSoup(rv.data, 'html.parser')
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
        rv = self.app.get('/models/images/generic/new')
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        assert 'New Image Model' in rv.data, 'unexpected page format'

    def test_nonexistent_model(self):
        assert not self.model_exists('foo'), "model shouldn't exist"

    def test_visualize_network(self):
        rv = self.app.post('/models/visualize-network?framework='+self.FRAMEWORK,
                data = {'custom_network': self.network()}
                )
        s = BeautifulSoup(rv.data, 'html.parser')
        body = s.select('body')
        if rv.status_code != 200:
            body = s.select('body')[0]
            if 'InvocationException' in str(body):
                raise unittest.SkipTest('GraphViz not installed')
            raise AssertionError('POST failed with %s\n\n%s' % (rv.status_code, body))
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

    def infer_one_for_job(self, job_id):
        # carry out one inference test per category in dataset
        image_path = os.path.join(self.imageset_folder, self.test_image)
        with open(image_path,'rb') as infile:
            # StringIO wrapping is needed to simulate POST file upload.
            image_upload = (StringIO(infile.read()), 'image.png')

        rv = self.app.post(
                '/models/images/generic/infer_one?job_id=%s' % job_id,
                data = {
                    'image_file': image_upload,
                    'show_visualizations': 'y',
                    }
                )
        s = BeautifulSoup(rv.data, 'html.parser')
        body = s.select('body')
        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)

    def test_infer_one_mean_image(self):
        # test the creation
        job_id = self.create_model(use_mean = 'image')
        assert self.model_wait_completion(job_id) == 'Done', 'job failed'
        self.infer_one_for_job(job_id)

    def test_infer_one_mean_pixel(self):
        # test the creation
        job_id = self.create_model(use_mean = 'pixel')
        assert self.model_wait_completion(job_id) == 'Done', 'job failed'
        self.infer_one_for_job(job_id)

    def test_infer_one_mean_none(self):
        # test the creation
        job_id = self.create_model(use_mean = 'none')
        assert self.model_wait_completion(job_id) == 'Done', 'job failed'
        self.infer_one_for_job(job_id)

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

        job2_id = self.create_model(**options)
        assert self.model_wait_completion(job2_id) == 'Done', 'second job failed'

    def test_retrain_twice(self):
        # retrain from a job which already had a pretrained model
        job1_id = self.create_model()
        assert self.model_wait_completion(job1_id) == 'Done', 'first job failed'
        rv = self.app.get('/models/%s.json' % job1_id)
        assert rv.status_code == 200, 'json load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        assert len(content['snapshots']), 'should have at least snapshot'
        options_2 = {
                'method': 'previous',
                'previous_networks': job1_id,
                }
        options_2['%s-snapshot' % job1_id] = content['snapshots'][-1]
        job2_id = self.create_model(**options_2)
        assert self.model_wait_completion(job2_id) == 'Done', 'second job failed'
        options_3 = {
                'method': 'previous',
                'previous_networks': job2_id,
                }
        options_3['%s-snapshot' % job2_id] = -1
        job3_id = self.create_model(**options_3)
        assert self.model_wait_completion(job3_id) == 'Done', 'third job failed'

    def test_diverging_network(self):
        if self.FRAMEWORK == 'caffe':
            raise unittest.SkipTest('Test not implemented for Caffe')
        job_id = self.create_model(json=True, learning_rate=1e15)
        assert self.model_wait_completion(job_id) == 'Error', 'job should have failed'
        job_info = self.job_info_html(job_id=job_id, job_type='models')
        assert 'Try decreasing your learning rate' in job_info

    def test_clone(self):
        options_1 = {
            'shuffle': True,
            'lr_step_size': 33.0,
            'previous_networks': 'None',
            'lr_inv_power': 0.5,
            'lr_inv_gamma': 0.1,
            'lr_poly_power': 3.0,
            'lr_exp_gamma': 0.95,
            'use_mean': 'image',
            'custom_network_snapshot': '',
            'lr_multistep_gamma': 0.5,
            'lr_policy': 'step',
            'crop_size': None,
            'val_interval': 3.0,
            'random_seed': 123,
            'learning_rate': 0.01,
            'standard_networks': 'None',
            'lr_step_gamma': 0.1,
            'lr_sigmoid_step': 50.0,
            'lr_sigmoid_gamma': 0.1,
            'lr_multistep_values': '50,85',
            'solver_type': 'SGD',
        }

        job1_id = self.create_model(**options_1)
        assert self.model_wait_completion(job1_id) == 'Done', 'first job failed'
        rv = self.app.get('/models/%s.json' % job1_id)
        assert rv.status_code == 200, 'json load failed with %s' % rv.status_code
        content1 = json.loads(rv.data)

        ## Clone job1 as job2
        options_2 = {
            'clone': job1_id,
        }

        job2_id = self.create_model(**options_2)
        assert self.model_wait_completion(job2_id) == 'Done', 'second job failed'
        rv = self.app.get('/models/%s.json' % job2_id)
        assert rv.status_code == 200, 'json load failed with %s' % rv.status_code
        content2 = json.loads(rv.data)

        ## These will be different
        content1.pop('id')
        content2.pop('id')
        content1.pop('directory')
        content2.pop('directory')
        assert (content1 == content2), 'job content does not match'

        job1 = digits.webapp.scheduler.get_job(job1_id)
        job2 = digits.webapp.scheduler.get_job(job2_id)

        assert (job1.form_data == job2.form_data), 'form content does not match'

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

    def test_models_page(self):
        rv = self.app.get('/models', follow_redirects=True)
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        assert 'Models' in rv.data, 'unexpected page format'

    def test_model_json(self):
        rv = self.app.get('/models/%s.json' % self.model_id)
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        assert content['id'] == self.model_id, 'expected different job_id'
        assert len(content['snapshots']) > 0, 'no snapshots in list'

    def test_edit_name(self):
        status = self.edit_job(
                self.dataset_id,
                name='new name'
                )
        assert status == 200, 'failed with %s' % status

    def test_edit_notes(self):
        status = self.edit_job(
                self.dataset_id,
                notes='new notes'
                )
        assert status == 200, 'failed with %s' % status

    def test_infer_one(self):
        image_path = os.path.join(self.imageset_folder, self.test_image)
        with open(image_path,'rb') as infile:
            # StringIO wrapping is needed to simulate POST file upload.
            image_upload = (StringIO(infile.read()), 'image.png')

        rv = self.app.post(
                '/models/images/generic/infer_one?job_id=%s' % self.model_id,
                data = {
                    'image_file': image_upload,
                    'show_visualizations': 'y',
                    }
                )
        s = BeautifulSoup(rv.data, 'html.parser')
        body = s.select('body')
        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)

    def test_infer_one_json(self):
        image_path = os.path.join(self.imageset_folder, self.test_image)
        with open(image_path,'rb') as infile:
            # StringIO wrapping is needed to simulate POST file upload.
            image_upload = (StringIO(infile.read()), 'image.png')

        rv = self.app.post(
                '/models/images/generic/infer_one.json?job_id=%s' % self.model_id,
                data = {
                    'image_file': image_upload,
                    }
                )
        assert rv.status_code == 200, 'POST failed with %s' % rv.status_code
        data = json.loads(rv.data)
        assert data['outputs']['output'][0][0] > 0 and \
                data['outputs']['output'][0][1] > 0, \
                'image regression result is wrong: %s' % data['outputs']['output']

    def test_infer_many(self):
        textfile_images = '%s\n' % self.test_image

        # StringIO wrapping is needed to simulate POST file upload.
        file_upload = (StringIO(textfile_images), 'images.txt')

        rv = self.app.post(
                '/models/images/generic/infer_many?job_id=%s' % self.model_id,
                data = {'image_list': file_upload}
                )
        s = BeautifulSoup(rv.data, 'html.parser')
        body = s.select('body')
        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)
        headers = s.select('table.table th')
        assert headers is not None, 'unrecognized page format'

    def test_infer_many_json(self):
        textfile_images = '%s\n' % self.test_image

        # StringIO wrapping is needed to simulate POST file upload.
        file_upload = (StringIO(textfile_images), 'images.txt')

        rv = self.app.post(
                '/models/images/generic/infer_many.json?job_id=%s' % self.model_id,
                data = {'image_list': file_upload}
                )
        assert rv.status_code == 200, 'POST failed with %s' % rv.status_code
        data = json.loads(rv.data)
        assert 'outputs' in data, 'invalid response'


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


class BaseTestCreatedCropInNetwork(BaseTestCreated):
    CAFFE_NETWORK = \
"""
layer {
  name: "data"
  type: "Data"
  top: "data"
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
  include {
    phase: TEST
  }
  transform_param {
    crop_size: 8
  }
}
layer {
  name: "scale"
  type: "Power"
  bottom: "data"
  top: "scale"
  power_param {
    scale: 0.004
  }
}
layer {
  name: "hidden"
  type: "InnerProduct"
  bottom: "scale"
  top: "output"
  inner_product_param {
    num_output: 2
  }
}
layer {
  name: "train_loss"
  type: "EuclideanLoss"
  bottom: "output"
  bottom: "label"
  top: "loss"
}
"""

    TORCH_NETWORK = \
"""
return function(p)
    local croplen = 8, channels
    if p.inputShape then channels=p.inputShape[1] else channels=1 end
    local net = nn.Sequential()
    net:add(nn.MulConstant(0.004))
    net:add(nn.View(-1):setNumInputDims(3))  -- flatten
    net:add(nn.Linear(channels*croplen*croplen,2)) -- c*croplen*croplen -> 2
    return {
        model = net,
        loss = nn.MSECriterion(),
        croplen = croplen
    }
end
"""

class BaseTestCreatedCropInForm(BaseTestCreated):
    CROP_SIZE = 8

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

class TestCaffeCreatedCropInNetwork(BaseTestCreatedCropInNetwork):
    FRAMEWORK = 'caffe'

class TestCaffeCreatedCropInForm(BaseTestCreatedCropInForm):
    FRAMEWORK = 'caffe'

class TestTorchViews(BaseTestViews):
    FRAMEWORK = 'torch'

class TestTorchCreation(BaseTestCreation):
    FRAMEWORK = 'torch'

class TestTorchCreated(BaseTestCreated):
    LR_POLICY = 'fixed'
    TRAIN_EPOCHS = 10
    FRAMEWORK = 'torch'

class TestTorchCreatedCropInNetwork(BaseTestCreatedCropInNetwork):
    LR_POLICY = 'fixed'
    TRAIN_EPOCHS = 10
    FRAMEWORK = 'torch'

class TestTorchCreatedCropInForm(BaseTestCreatedCropInForm):
    LR_POLICY = 'fixed'
    TRAIN_EPOCHS = 10
    FRAMEWORK = 'torch'

class TestTorchDatasetModelInteractions(BaseTestDatasetModelInteractions):
    FRAMEWORK = 'torch'
