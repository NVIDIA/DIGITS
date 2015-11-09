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
import tempfile

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
import digits.dataset.images.classification.test_views
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
return function(p)
    -- model should adjust to any 3D input
    local nDim = 1
    if p.inputShape then p.inputShape:apply(function(x) nDim=nDim*x end) end
    local model = nn.Sequential()
    model:add(nn.View(-1):setNumInputDims(3)) -- c*h*w -> chw (flattened)
    model:add(nn.Linear(nDim, 3)) -- chw -> 3
    model:add(nn.LogSoftMax())
    return {
        model = model
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
        digits.dataset.images.classification.test_views.BaseViewsTestWithDataset):
    """
    Provides a dataset
    """

    # Inherited classes may want to override these attributes
    CROP_SIZE = None
    TRAIN_EPOCHS = 1
    SHUFFLE = False
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
    def create_model(cls, network=None, **kwargs):
        """
        Create a model
        Returns the job_id
        Raise RuntimeError if job fails to create

        Keyword arguments:
        **kwargs -- data to be sent with POST request
        """
        if network is None:
            network = cls.network()
        data = {
                'model_name':       'test_model',
                'dataset':          cls.dataset_id,
                'method':           'custom',
                'custom_network':   network,
                'batch_size':       10,
                'train_epochs':     cls.TRAIN_EPOCHS,
                'framework' :       cls.FRAMEWORK,
                'random_seed':      0xCAFEBABE,
                'shuffle':          'true' if cls.SHUFFLE else 'false'
                }
        if cls.CROP_SIZE is not None:
            data['crop_size'] = cls.CROP_SIZE
        if cls.LR_POLICY is not None:
            data['lr_policy'] = cls.LR_POLICY
        if cls.LEARNING_RATE is not None:
            data['learning_rate'] = cls.LEARNING_RATE
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
        rv = self.app.get('/models/images/classification/new')
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        assert 'New Image Classification Model' in rv.data, 'unexpected page format'

    def test_nonexistent_model(self):
        assert not self.model_exists('foo'), "model shouldn't exist"

    def test_visualize_network(self):
        rv = self.app.post('/models/visualize-network?framework='+self.FRAMEWORK,
                data = {'custom_network': self.network()}
                )
        s = BeautifulSoup(rv.data, 'html.parser')
        if rv.status_code != 200:
            body = s.select('body')[0]
            if 'InvocationException' in str(body):
                raise unittest.SkipTest('GraphViz not installed')
            raise AssertionError('POST failed with %s\n\n%s' % (rv.status_code, body))
        image = s.select('img')
        assert image is not None, "didn't return an image"

    def test_customize(self):
        rv = self.app.post('/models/customize?network=lenet&framework='+self.FRAMEWORK)
        s = BeautifulSoup(rv.data, 'html.parser')
        body = s.select('body')
        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)

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
                if self.FRAMEWORK=='torch' and len(combination)>1:
                    raise unittest.SkipTest('Torch not tested with multi-GPU')
                yield self.check_select_gpus, combination

    def check_select_gpus(self, gpu_list):
        job_id = self.create_model(select_gpus_list=','.join(gpu_list), batch_size=len(gpu_list))
        assert self.model_wait_completion(job_id) == 'Done', 'create failed'

    def classify_one_for_job(self, job_id, test_misclassification = True):
        # carry out one inference test per category in dataset
        for category in self.imageset_paths.keys():
            image_path = self.imageset_paths[category][0]
            image_path = os.path.join(self.imageset_folder, image_path)
            with open(image_path,'rb') as infile:
                # StringIO wrapping is needed to simulate POST file upload.
                image_upload = (StringIO(infile.read()), 'image.png')

            rv = self.app.post(
                    '/models/images/classification/classify_one?job_id=%s' % job_id,
                    data = {
                        'image_file': image_upload,
                        }
                    )
            s = BeautifulSoup(rv.data, 'html.parser')
            body = s.select('body')
            assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)
            # gets an array of arrays [[confidence, label],...]
            predictions = [p.get_text().split() for p in s.select('ul.list-group li')]
            if test_misclassification:
                assert predictions[0][1] == category, 'image misclassified'

    def test_classify_one_mean_image(self):
        # test the creation
        job_id = self.create_model(use_mean = 'image')
        assert self.model_wait_completion(job_id) == 'Done', 'job failed'
        self.classify_one_for_job(job_id)

    def test_classify_one_mean_pixel(self):
        # test the creation
        job_id = self.create_model(use_mean = 'pixel')
        assert self.model_wait_completion(job_id) == 'Done', 'job failed'
        self.classify_one_for_job(job_id)

    def test_classify_one_mean_none(self):
        # test the creation
        job_id = self.create_model(use_mean = 'none')
        assert self.model_wait_completion(job_id) == 'Done', 'job failed'
        self.classify_one_for_job(job_id, False)

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

    def test_bad_network_definition(self):
        if self.FRAMEWORK == 'caffe':
            bogus_net = """
                layer {
                    name: "hidden"
                    type: 'BogusCode'
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
                """
        elif self.FRAMEWORK == 'torch':
            bogus_net = """
                local model = BogusCode(0)
                return function(params)
                    return {
                        model = model
                    }
                end
                """
        job_id = self.create_model(json=True, network=bogus_net)
        assert self.model_wait_completion(job_id) == 'Error', 'job should have failed'
        job_info = self.job_info_html(job_id=job_id, job_type='models')
        assert 'BogusCode' in job_info, "job_info: \n%s" % str(job_info)

    def test_clone(self):
        options_1 = {
            'shuffle': True,
            'snapshot_interval': 2.0,
            'lr_step_size': 33.0,
            'lr_inv_power': 0.5,
            'lr_inv_gamma': 0.1,
            'lr_poly_power': 3.0,
            'lr_exp_gamma': 0.9,
            'use_mean': 'image',
            'lr_multistep_gamma': 0.5,
            'lr_policy': 'exp',
            'val_interval': 3.0,
            'random_seed': 123,
            'learning_rate': 0.0125,
            'lr_step_gamma': 0.1,
            'lr_sigmoid_step': 50.0,
            'lr_sigmoid_gamma': 0.1,
            'lr_multistep_values': '50,85',
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
        assert content['id'] == self.model_id, 'id %s != %s' % (content['id'], self.model_id)
        assert content['dataset_id'] == self.dataset_id, 'dataset_id %s != %s' % (content['dataset_id'], self.dataset_id)
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

    def test_classify_one(self):
        # carry out one inference test per category in dataset
        for category in self.imageset_paths.keys():
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
            s = BeautifulSoup(rv.data, 'html.parser')
            body = s.select('body')
            assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)
            # gets an array of arrays [[confidence, label],...]
            predictions = [p.get_text().split() for p in s.select('ul.list-group li')]
            assert predictions[0][1] == category, 'image misclassified'

    def test_classify_one_json(self):
        for category in self.imageset_paths.keys():
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
        s = BeautifulSoup(rv.data, 'html.parser')
        body = s.select('body')
        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)

    def test_classify_many_invalid_ground_truth(self):
        textfile_images = ''
        label_id = 0
        for label, images in self.imageset_paths.iteritems():
            for image in images:
                image_path = image
                image_path = os.path.join(self.imageset_folder, image_path)
                # test label_id with -1 and >len(labels)
                textfile_images += '%s %s\n' % (image_path, 3*label_id-1)
            label_id += 1

        # StringIO wrapping is needed to simulate POST file upload.
        file_upload = (StringIO(textfile_images), 'images.txt')

        rv = self.app.post(
                '/models/images/classification/classify_many?job_id=%s' % self.model_id,
                data = {'image_list': file_upload}
                )
        s = BeautifulSoup(rv.data, 'html.parser')
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
        s = BeautifulSoup(rv.data, 'html.parser')
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
    TORCH_NETWORK = \
"""
return function(p)
    local croplen = 8, channels
    if p.inputShape then channels=p.inputShape[1] else channels=1 end
    local model = nn.Sequential()
    model:add(nn.View(-1):setNumInputDims(3)) -- flatten
    model:add(nn.Linear(channels*croplen*croplen, 3)) -- chw -> 3
    model:add(nn.LogSoftMax())
    return {
        model = model,
        croplen = croplen
    }
end
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
    TRAIN_EPOCHS = 10

class TestTorchCreated(BaseTestCreated):
    FRAMEWORK = 'torch'
    TRAIN_EPOCHS = 10

class TestTorchCreatedShuffle(TestTorchCreated):
    SHUFFLE = True

class TestTorchCreatedHdf5(TestTorchCreated):
    BACKEND = 'hdf5'

class TestTorchCreatedHdf5Shuffle(TestTorchCreatedHdf5):
    SHUFFLE = True

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

class TestTorchCreatedCropInForm(BaseTestCreatedCropInForm):
    FRAMEWORK = 'torch'
    TRAIN_EPOCHS = 10

class TestTorchCreatedCropInNetwork(BaseTestCreatedCropInNetwork):
    FRAMEWORK = 'torch'
    TRAIN_EPOCHS = 10

class TestTorchCreatedWide(BaseTestCreatedWide):
    FRAMEWORK = 'torch'
    TRAIN_EPOCHS = 10

class TestTorchCreatedTall(BaseTestCreatedTall):
    FRAMEWORK = 'torch'
    TRAIN_EPOCHS = 10

class TestTorchLeNet(TestTorchCreated):
    IMAGE_WIDTH = 28
    IMAGE_HEIGHT = 28
    TRAIN_EPOCHS = 20
    # need more aggressive learning rate
    # on such a small dataset
    LR_POLICY = 'fixed'
    LEARNING_RATE = 0.1

    # standard lenet model will adjust to color
    # or grayscale images
    TORCH_NETWORK=open(
            os.path.join(
                os.path.dirname(digits.__file__),
                'standard-networks', 'torch', 'lenet.lua')
            ).read()

class TestTorchLeNetShuffle(TestTorchLeNet):
    SHUFFLE = True

class TestTorchLeNetHdf5(TestTorchLeNet):
    BACKEND = 'hdf5'

class TestTorchLeNetHdf5Shuffle(TestTorchLeNetHdf5):
    SHUFFLE = True

class TestPythonLayer(BaseViewsTestWithDataset):
    FRAMEWORK = 'caffe'
    CAFFE_NETWORK = """\
layer {
    name: "hidden"
    type: 'InnerProduct'
    inner_product_param {
        num_output: 500
        weight_filler {
            type: "xavier"
        }
        bias_filler {
             type: "constant"
        }
    }
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
layer {
    name: "py_test"
    type: "Python"
    bottom: "output"
    top: "py_test"
    python_param {
        module: "py_test"
        layer: "PythonLayer"
    }
}

"""
    def write_python_layer_script(self, filename):
        with open(filename, 'w') as f:
            f.write("""\
import caffe
import numpy as np

class PythonLayer(caffe.Layer):

    def setup(self, bottom, top):
        print 'PythonLayer::setup'
        if len(bottom) != 1:
            raise Exception("Need one input.")

    def reshape(self, bottom, top):
        print 'PythonLayer::reshape'
        top[0].reshape(1)

    def forward(self, bottom, top):
        print 'PythonLayer::forward'
        top[0].data[...] = np.sum(bottom[0].data) / 2. / bottom[0].num
""")

    ## This test makes a temporary python layer file whose path is set
    ## as py_layer_server_file.  The job creation process copies that
    ## file to the job_dir.  The CAFFE_NETWORK above, requires that
    ## python script to be in the correct spot. If there is an error
    ## in the script or if the script is named incorrectly, or does
    ## not exist in the job_dir, then the test will fail.
    def test_python_layer(self):
        tmpdir = tempfile.mkdtemp()
        py_file = tmpdir + '/py_test.py'
        self.write_python_layer_script(py_file)

        job_id = self.create_model(python_layer_server_file=py_file)

        # remove the temporary python script.
        shutil.rmtree(tmpdir)

        assert self.model_wait_completion(job_id) == 'Done', 'first job failed'
        rv = self.app.get('/models/%s.json' % job_id)
        assert rv.status_code == 200, 'json load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        assert len(content['snapshots']), 'should have at least snapshot'
