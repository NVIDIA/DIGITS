# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import json
import os
import shutil
import tempfile
import time
import unittest
import itertools

from gevent import monkey
monkey.patch_all()
from bs4 import BeautifulSoup
import numpy as np
import PIL.Image
from urlparse import urlparse
from cStringIO import StringIO

import webapp
from config import config_value
import device_query


DUMMY_IMAGE_DIM = 10
DUMMY_IMAGE_COUNT = 10 # per category

# TODO: these might be too short on a slow system
TIMEOUT_DATASET = 10
TIMEOUT_MODEL = 10

def create_dummy_dataset(data_path):
    """
    A very simple dataset - Red, Green and Blue PNGs
    """
    dim = DUMMY_IMAGE_DIM
    count = DUMMY_IMAGE_COUNT
    min_color = 200
    labels = {'red': 0, 'green': 1, 'blue': 2}
    # Stores the relative path of each image of the dataset.
    images = {'red': [], 'green': [], 'blue': []}
    for (label, idx) in labels.iteritems():
        label_path = label
        os.mkdir(os.path.join(data_path, label_path))

        colors = np.linspace(min_color, 255, count)
        for i in range(count):
            pixel = [0, 0, 0]
            pixel[idx] = colors[i]
            img = np.full((dim, dim, 3), pixel, dtype=np.uint8)
            pil_img = PIL.Image.fromarray(img)
            img_path = os.path.join(label_path, str(i) + '.png')
            pil_img.save(os.path.join(data_path, img_path))
            images[label].append(img_path)

    return images

def get_dummy_network():
    """
    A very simple network - one fully connected layer
    """
    return \
    """
    layer {
        name: "in"
        type: 'InnerProduct'
        bottom: "data"
        top: "in"
        inner_product_param {
            num_output: 3
        }
    }
    layer {
        name: "loss"
        type: "SoftmaxWithLoss"
        bottom: "in"
        bottom: "label"
        top: "loss"
    }
    layer {
        name: "accuracy"
        type: "Accuracy"
        bottom: "in"
        bottom: "label"
        top: "accuracy"
        include {
            phase: TEST
        }
    }
    """


class WebappBaseTest(object):
    """
    Defines some methods useful across the different webapp test suites
    """
    @classmethod
    def setUpClass(cls):
        # Create some dummy data
        cls.data_path = tempfile.mkdtemp()
        cls.images = create_dummy_dataset(cls.data_path)
        # Start up the server
        assert webapp.scheduler.start(), "scheduler wouldn't start"
        webapp.app.config['WTF_CSRF_ENABLED'] = False
        webapp.app.config['TESTING'] = True
        cls.app = webapp.app.test_client()
        cls.created_datasets = []
        cls.created_models = []

    @classmethod
    def tearDownClass(cls):
        # Remove all jobs
        for job_id in cls.created_models:
            cls.delete_model(job_id)
        for job_id in cls.created_datasets:
            cls.delete_dataset(job_id)
        # Remove the dummy data
        shutil.rmtree(cls.data_path)

    @classmethod
    def create_dataset(cls, **data):
        """
        Create a dataset
        Returns the job_id
        Raises RuntimeError if job fails to create

        Arguments:
        data -- data to be sent with POST request
        """
        if 'dataset_name' not in data:
            data['dataset_name'] = 'dummy_dataset'

        request_json = data.pop('json', False)
        url = '/datasets/images/classification'
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
                raise RuntimeError('Failed to create dataset')

        job_id = cls.job_id_from_response(rv)

        assert cls.dataset_exists(job_id), 'dataset not found after successful creation'

        cls.created_datasets.append(job_id)
        return job_id

    @classmethod
    def create_quick_dataset(cls, **kwargs):
        """
        Creates a simple dataset quickly
        Returns the job_id

        Keyword arguments:
        kwargs -- any overrides you want to pass into the POST data
        """
        defaults = {
                'method': 'folder',
                'folder_train': cls.data_path,
                'resize_width': DUMMY_IMAGE_DIM,
                'resize_height': DUMMY_IMAGE_DIM,
                }
        defaults.update(kwargs)
        return cls.create_dataset(**defaults)

    @classmethod
    def create_model(cls, **data):
        """
        Create a model
        Returns the job_id
        Raise RuntimeError if job fails to create

        Arguments:
        data -- data to be sent with POST request
        """
        if 'model_name' not in data:
            data['model_name'] = 'dummy_model'

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

    @classmethod
    def create_quick_model(cls, dataset_id, **kwargs):
        """
        Creates a simple model quickly
        Returns the job_id

        Arguments:
        dataset_id -- id for the dataset

        Keyword arguments:
        kwargs -- any overrides you want to pass into the POST data
        """
        defaults = {
                'dataset': dataset_id,
                'method': 'custom',
                'custom_network': get_dummy_network(),
                'batch_size': DUMMY_IMAGE_COUNT,
                'train_epochs': 1,
                }
        defaults.update(kwargs)
        return cls.create_model(**defaults)

    @classmethod
    def job_id_from_response(cls, rv):
        """
        Extract the job_id from an HTTP response
        """
        job_url = rv.headers['Location']
        parsed_url = urlparse(job_url)
        return parsed_url.path.split('/')[-1]

    @classmethod
    def dataset_exists(cls, job_id):
        return cls.job_exists(job_id, 'datasets')

    @classmethod
    def model_exists(cls, job_id):
        return cls.job_exists(job_id, 'models')

    @classmethod
    def job_exists(cls, job_id, job_type='jobs'):
        """
        Test whether a job exists
        """
        url = '/%s/%s' % (job_type, job_id)
        rv = cls.app.get(url, follow_redirects=True)
        assert rv.status_code in [200, 404], 'got status code "%s" from "%s"' % (rv.status_code, url)
        return rv.status_code == 200

    @classmethod
    def dataset_status(cls, job_id):
        return cls.job_status(job_id, 'datasets')

    @classmethod
    def model_status(cls, job_id):
        return cls.job_status(job_id, 'models')

    @classmethod
    def job_status(cls, job_id, job_type='jobs'):
        """
        Get the status of a job
        """
        url = '/%s/%s/status' % (job_type, job_id)
        rv = cls.app.get(url)
        assert rv.status_code == 200, 'Cannot get status of job %s. "%s" returned %s' % (job_id, url, rv.status_code)
        status = json.loads(rv.data)
        return status['status']

    @classmethod
    def abort_dataset(cls, job_id):
        return cls.abort_job(job_id, job_type='datasets')

    @classmethod
    def abort_model(cls, job_id):
        return cls.abort_job(job_id, job_type='models')

    @classmethod
    def abort_job(cls, job_id, job_type='jobs'):
        """
        Abort a job
        Returns the HTTP status code
        """
        rv = cls.app.post('/%s/%s/abort' % (job_type, job_id))
        return rv.status_code

    @classmethod
    def dataset_wait_completion(cls, job_id, **kwargs):
        kwargs['job_type'] = 'datasets'
        if 'timeout' not in kwargs:
            kwargs['timeout'] = TIMEOUT_DATASET
        return cls.job_wait_completion(job_id, **kwargs)

    @classmethod
    def model_wait_completion(cls, job_id, **kwargs):
        kwargs['job_type'] = 'models'
        if 'timeout' not in kwargs:
            kwargs['timeout'] = TIMEOUT_MODEL
        return cls.job_wait_completion(job_id, **kwargs)

    @classmethod
    def job_wait_completion(cls, job_id, timeout=10, polling_period=0.5, job_type='jobs'):
        """
        Poll the job status until it completes
        Returns the final status

        Arguments:
        job_id -- the job to wait for

        Keyword arguments:
        timeout -- maximum wait time (seconds)
        polling_period -- how often to poll (seconds)
        job_type -- [datasets|models]
        """
        start = time.time()
        while True:
            status = cls.job_status(job_id, job_type=job_type)
            if status in ['Done', 'Abort', 'Error']:
                return status
            assert (time.time() - start) < timeout, 'Job took more than %s seconds' % timeout
            time.sleep(polling_period)

    @classmethod
    def delete_dataset(cls, job_id):
        return cls.delete_job(job_id, job_type='datasets')

    @classmethod
    def delete_model(cls, job_id):
        return cls.delete_job(job_id, job_type='models')

    @classmethod
    def delete_job(cls, job_id, job_type='jobs'):
        """
        Delete a job
        Returns the HTTP status code
        """
        rv = cls.app.delete('/%s/%s' % (job_type, job_id))
        return rv.status_code

################################################################################
# Tests start here
################################################################################

class TestWebapp(WebappBaseTest):
    """
    Some app-wide tests
    """
    def test_page_home(self):
        """home page"""
        rv = self.app.get('/')
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        for h in ['Home', 'Datasets', 'Models']:
            assert h in rv.data, 'unexpected page format'

    def test_invalid_page(self):
        """invalid page"""
        rv = self.app.get('/foo')
        assert rv.status_code == 404, 'should return 404'

    def test_invalid_dataset(self):
        """invalid dataset"""
        assert not self.dataset_exists('foo'), "dataset shouldn't exist"

    def test_invalid_model(self):
        """invalid model"""
        assert not self.model_exists('foo'), "model shouldn't exist"


class TestDatasetCreation(WebappBaseTest):
    """
    Dataset creation tests
    """
    def test_page_dataset_new(self):
        """new image classification dataset page"""
        rv = self.app.get('/datasets/images/classification/new')
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        assert 'New Image Classification Dataset' in rv.data, 'unexpected page format'

    def test_invalid_folder(self):
        """invalid folder"""
        empty_dir = tempfile.mkdtemp()
        try:
            job_id = self.create_dataset(
                    method = 'folder',
                    train_folder = empty_dir
                    )
        except RuntimeError:
            return
        raise AssertionError('Should have failed')

    def test_create_json(self):
        """dataset - create w/ json"""
        self.create_quick_dataset(json=True)

    def test_create_delete(self):
        """dataset - create, delete"""
        job_id = self.create_quick_dataset()
        assert self.delete_dataset(job_id) == 200, 'delete failed'
        assert not self.dataset_exists(job_id), 'dataset exists after delete'

    def test_create_wait_delete(self):
        """dataset - create, wait, delete"""
        job_id = self.create_quick_dataset()
        assert self.dataset_wait_completion(job_id) == 'Done', 'create failed'
        assert self.delete_dataset(job_id) == 200, 'delete failed'
        assert not self.dataset_exists(job_id), 'dataset exists after delete'

    def test_create_abort_delete(self):
        """dataset - create, abort, delete"""
        job_id = self.create_quick_dataset()
        assert self.abort_dataset(job_id) == 200, 'abort failed'
        assert self.delete_dataset(job_id) == 200, 'delete failed'
        assert not self.dataset_exists(job_id), 'dataset exists after delete'


    def create_from_textfiles(self, absolute_path=True):
        """
        Create a dataset from textfiles

        Arguments:
        absolute_path -- if False, give relative paths and image folders
        """
        textfile_train_images = ''
        textfile_labels_file = ''
        label_id = 0
        for (label, images) in self.images.iteritems():
            textfile_labels_file += '%s\n' % label
            for image in images:
                image_path = image
                if absolute_path:
                    image_path = os.path.join(self.data_path, image_path)
                textfile_train_images += '%s %d\n' % (image_path, label_id)

            label_id += 1

        # StringIO wrapping is needed to simulate POST file upload.
        train_upload = (StringIO(textfile_train_images), 'train.txt')
        # Use the same list for training and validation.
        val_upload = (StringIO(textfile_train_images), 'val.txt')
        labels_upload = (StringIO(textfile_labels_file), 'labels.txt')

        data = {
                'method': 'textfile',
                'textfile_train_images': train_upload,
                'textfile_use_val': 'y',
                'textfile_val_images': val_upload,
                'textfile_labels_file': labels_upload,
                }
        if not absolute_path:
            data['textfile_train_folder'] = self.data_path
            data['textfile_val_folder'] = self.data_path

        return self.create_dataset(**data)

    def test_textfile_absolute(self):
        """dataset - textfiles (absolute), wait"""
        job_id = self.create_from_textfiles(absolute_path=True)
        assert self.dataset_wait_completion(job_id) == 'Done', 'create failed'

    def test_textfile_relative(self):
        """dataset - textfiles (relative), wait"""
        job_id = self.create_from_textfiles(absolute_path=False)
        status = self.dataset_wait_completion(job_id)
        assert status == 'Done', 'create failed "%s"' % status

    def test_nonsquare_dimensions(self):
        """dataset - nonsquare dimensions"""
        job_id = self.create_quick_dataset(
                resize_width = DUMMY_IMAGE_DIM,
                resize_height = DUMMY_IMAGE_DIM*2,
                )
        status = self.dataset_wait_completion(job_id)
        assert status == 'Done', 'create failed "%s"' % status
        img_url = '/files/%s/mean.jpg' % job_id
        rv = self.app.get(img_url)
        assert rv.status_code == 200, 'GET on %s returned %s' % (img_url, rv.status_code)
        buff = StringIO(rv.data)
        buff.seek(0)
        size = PIL.Image.open(buff).size
        assert size == (DUMMY_IMAGE_DIM,DUMMY_IMAGE_DIM*2), 'image size is %s' % (size,)

class TestCreatedDataset(WebappBaseTest):
    """
    Tests on a dataset that has already been created
    """
    @classmethod
    def setUpClass(cls):
        super(TestCreatedDataset, cls).setUpClass()
        cls.dataset_id = cls.create_quick_dataset()
        assert cls.dataset_wait_completion(cls.dataset_id) == 'Done', 'dataset creation failed'

    def test_index_json(self):
        """created dataset - index.json"""
        rv = self.app.get('/index.json')
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        found = False
        for d in content['datasets']:
            if d['id'] == self.dataset_id:
                found = True
                break
        assert found, 'dataset not found in list'

    def test_dataset_json(self):
        """created dataset - json"""
        rv = self.app.get('/datasets/%s.json' % self.dataset_id)
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        assert content['id'] == self.dataset_id, 'expected different job_id'

class TestModelCreation(WebappBaseTest):
    """
    Model creation tests
    """
    @classmethod
    def setUpClass(cls):
        super(TestModelCreation, cls).setUpClass()
        cls.dataset_id = cls.create_dataset(
                method = 'folder',
                folder_train = cls.data_path,
                resize_width = DUMMY_IMAGE_DIM,
                resize_height = DUMMY_IMAGE_DIM,
                )

    def test_page_model_new(self):
        """new image classification model page"""
        rv = self.app.get('/models/images/classification/new')
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        assert 'New Image Classification Model' in rv.data, 'unexpected page format'

    def test_visualize_network(self):
        """visualize network"""
        rv = self.app.post('/models/visualize-network',
                data = {'custom_network': get_dummy_network()}
                )
        s = BeautifulSoup(rv.data)
        body = s.select('body')
        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)
        image = s.select('img')
        assert image is not None, "didn't return an image"

    def test_create_json(self):
        """model - create w/ json"""
        self.create_quick_model(self.dataset_id, json=True)

    def test_create_delete(self):
        """model - create, delete"""
        job_id = self.create_quick_model(self.dataset_id)
        assert self.delete_model(job_id) == 200, 'delete failed'
        assert not self.model_exists(job_id), 'model exists after delete'

    def test_create_wait_delete(self):
        """model - create, wait, delete"""
        job_id = self.create_quick_model(self.dataset_id)
        assert self.model_wait_completion(job_id) == 'Done', 'create failed'
        assert self.delete_model(job_id) == 200, 'delete failed'
        assert not self.model_exists(job_id), 'model exists after delete'

    def test_create_abort_delete(self):
        """model - create, abort, delete"""
        job_id = self.create_quick_model(self.dataset_id)
        assert self.abort_model(job_id) == 200, 'abort failed'
        assert self.delete_model(job_id) == 200, 'delete failed'
        assert not self.model_exists(job_id), 'model exists after delete'

    def test_snapshot_interval_2(self):
        """model - snapshot_interval 2"""
        job_id = self.create_quick_model(self.dataset_id, train_epochs=1, snapshot_interval=0.5)
        assert self.model_wait_completion(job_id) == 'Done', 'create failed'
        rv = self.app.get('/models/%s.json' % job_id)
        assert rv.status_code == 200, 'json load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        assert len(content['snapshots']) > 1, 'should take >1 snapshot'

    def test_snapshot_interval_0_5(self):
        """model - snapshot_interval 0.5"""
        job_id = self.create_quick_model(self.dataset_id, train_epochs=4, snapshot_interval=2)
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
        """model - select GPU"""
        for index in config_value('gpu_list').split(','):
            yield self.check_select_gpu, index

    def check_select_gpu(self, gpu_index):
        job_id = self.create_quick_model(self.dataset_id, select_gpu=gpu_index)
        assert self.delete_model(job_id) == 200, 'delete failed'

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
        """model - select GPUs"""
        # test all possible combinations
        gpu_list = config_value('gpu_list').split(',')
        for i in xrange(len(gpu_list)):
            for combination in itertools.combinations(gpu_list, i+1):
                yield self.check_select_gpus, combination

    def check_select_gpus(self, gpu_list):
        job_id = self.create_quick_model(self.dataset_id,
                select_gpus_list=','.join(gpu_list))
        assert self.delete_model(job_id) == 200, 'delete failed'

class TestCreatedModel(WebappBaseTest):
    """
    Tests on a model that has already been created
    """
    @classmethod
    def setUpClass(cls):
        super(TestCreatedModel, cls).setUpClass()
        cls.dataset_id = cls.create_quick_dataset()
        assert cls.dataset_wait_completion(cls.dataset_id) == 'Done', 'dataset creation failed'
        cls.model_id = cls.create_quick_model(cls.dataset_id)
        assert cls.model_wait_completion(cls.model_id) == 'Done', 'model creation failed'

    def download_model(self, extension):
        url = '/models/%s/download.%s' % (self.model_id, extension)
        rv = self.app.get(url)
        assert rv.status_code == 200, 'download "%s" failed with %s' % (url, rv.status_code)

    def test_download(self):
        """created model - download"""
        for extension in ['tar', 'zip', 'tar.gz', 'tar.bz2']:
            yield self.download_model, extension

    def test_index_json(self):
        """created model - index.json"""
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
        """created model - json"""
        rv = self.app.get('/models/%s.json' % self.model_id)
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        assert content['id'] == self.model_id, 'expected different job_id'
        assert len(content['snapshots']) > 0, 'no snapshots in list'

    def test_classify_one(self):
        """created model - classify one"""
        image_path = self.images['green'][0]
        image_path = os.path.join(self.data_path, image_path)
        with open(image_path) as infile:
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
        assert predictions[0][1] == 'green', 'image misclassified'

    def test_classify_one_json(self):
        """created model - classify one JSON"""
        image_path = self.images['green'][0]
        image_path = os.path.join(self.data_path, image_path)
        with open(image_path) as infile:
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
        assert data['predictions'][0][0] == 'green', 'image misclassified'

    def test_classify_many(self):
        """created model - classify many"""
        textfile_images = ''
        label_id = 0
        for (label, images) in self.images.iteritems():
            for image in images:
                image_path = image
                image_path = os.path.join(self.data_path, image_path)
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
        """created model - classify many JSON"""
        textfile_images = ''
        label_id = 0
        for (label, images) in self.images.iteritems():
            for image in images:
                image_path = image
                image_path = os.path.join(self.data_path, image_path)
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
        """created model - top n predictions"""
        textfile_images = ''
        label_id = 0
        for (label, images) in self.images.iteritems():
            for image in images:
                image_path = image
                image_path = os.path.join(self.data_path, image_path)
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
        keys = self.images.keys()
        for key in keys:
            assert key in rv.data, '"%s" not found in the response'

    def test_retrain(self):
        """created model - retrain"""
        options = {}
        options['previous_networks'] = self.model_id
        rv = self.app.get('/models/%s.json' % self.model_id)
        assert rv.status_code == 200, 'json load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        assert len(content['snapshots']), 'should have at least snapshot'
        options['%s-snapshot' % self.model_id] = content['snapshots'][-1]
        job_id = self.create_quick_model(self.dataset_id,
                method='previous', **options)
        self.abort_model(job_id)

class TestDatasetModelInteractions(WebappBaseTest):
    """
    Test the interactions between datasets and models
    """

    def test_model_with_deleted_database(self):
        """model on deleted dataset"""
        dataset_id = self.create_quick_dataset()
        assert self.delete_dataset(dataset_id) == 200, 'delete failed'
        assert not self.dataset_exists(dataset_id), 'dataset exists after delete'

        try:
            model_id = self.create_quick_model(dataset_id)
        except RuntimeError:
            return
        assert False, 'Should have failed'

    def test_model_on_running_dataset(self):
        """model on running dataset"""
        dataset_id = self.create_quick_dataset()
        model_id = self.create_quick_model(dataset_id)
        # should wait until dataset has finished
        assert self.model_status(model_id) in ['Initialized', 'Waiting', 'Done'], 'model not waiting'
        assert self.dataset_wait_completion(dataset_id) == 'Done', 'dataset creation failed'
        time.sleep(1)
        # then it should start
        assert self.model_status(model_id) in ['Running', 'Done'], "model didn't start"
        self.abort_model(model_id)

    # A dataset should not be deleted while a model using it is running.
    def test_model_create_dataset_delete(self):
        """delete dataset with dependent model"""
        dataset_id = self.create_quick_dataset()
        model_id = self.create_quick_model(dataset_id)
        assert self.dataset_wait_completion(dataset_id) == 'Done', 'dataset creation failed'
        assert self.delete_dataset(dataset_id) == 403, 'dataset deletion should not have succeeded'
        self.abort_model(model_id)

