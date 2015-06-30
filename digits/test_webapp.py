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

# May be too short on a slow system
TIMEOUT_DATASET = 15
TIMEOUT_MODEL = 20

class Dataset():
    def __init__(self, data_path, images):
        self.data_path = data_path
        self.images = images

class Direction():
    RIGHT, UP, LEFT, DOWN = range(4)

class ImageType():
    TYPES = ['COLOR', 'GRAY']
    COLOR, GRAY = TYPES

LABELS = {
    'red-to-right': (0, Direction.RIGHT),
    'green-to-left': (1, Direction.LEFT),
    'blue-to-bottom': (2, Direction.DOWN)
}

CATEGORIES = LABELS.keys()

dummy_network = \
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

def create_color_direction(size, color_from, color_to, direction):
    """
    Make an image with a color gradient in a specific direction
    """
    # create gradient
    rgb_arrays = [np.linspace(color_from[x], color_to[x], size).astype('uint8') for x in range(3)]
    gradient = np.concatenate(rgb_arrays)

    # extend to 2d
    picture = np.repeat(gradient, size)
    picture.shape = (3, size, size)

    # make image and rotate
    image = PIL.Image.fromarray(picture.T)
    image = image.rotate(direction*90)

    return image

def build_dataset(data_path):
    """
    A very simple dataset - Red, Green and Blue PNGs with gradients
    """
    dim = DUMMY_IMAGE_DIM
    count = DUMMY_IMAGE_COUNT
    min_color = 200

    # Stores the relative path of each image of the dataset.
    images = {k: [] for k in CATEGORIES}

    for (label, (idx, direction)) in LABELS.iteritems():
        label_path = label
        os.mkdir(os.path.join(data_path, label_path))

        colors = np.linspace(min_color, 255, count)
        for i in range(count):
            pixel = [0, 0, 0]
            pixel[idx] = colors[i]
            pil_img = create_color_direction(dim, (0, 0, 0), pixel, direction)
            img_path = os.path.join(label_path, str(i) + '.png')
            pil_img.save(os.path.join(data_path, img_path))
            images[label].append(img_path)

    return images

class WebappBaseTest(object):
    """
    Defines some methods useful across the different webapp test suites
    """
    @classmethod
    def setUpClass(cls):
        # Create some dummy data
        cls.image_type_data = {}
        for image_type in ImageType.TYPES:
            data_path = tempfile.mkdtemp()
            images = build_dataset(data_path)
            cls.image_type_data[image_type] = Dataset(data_path, images)

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
        [shutil.rmtree(x.data_path) for x in cls.image_type_data.itervalues()]

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
    def create_quick_dataset(cls, image_type, **kwargs):
        """
        Creates a simple dataset quickly
        Returns the job_id

        Keyword arguments:
        kwargs -- any overrides you want to pass into the POST data
        """
        if image_type is ImageType.GRAY:
            channels = 1
        else:
            channels = 3

        defaults = {
                'method': 'folder',
                'resize_channels': channels,
                'folder_train': cls.image_type_data[image_type].data_path,
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
                'custom_network': dummy_network,
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

    def test_alltests(self):
        for image_type in ImageType.TYPES:
            yield self.check_create_json, image_type
            yield self.check_create_delete, image_type
            yield self.check_create_wait_delete, image_type
            yield self.check_create_abort_delete, image_type
            yield self.check_nonsquare_dimensions, image_type

            for absolute_path in (True, False):
                yield self.check_textfile, image_type, absolute_path

    def check_create_json(self, image_type):
        """dataset - create w/ json"""
        self.create_quick_dataset(image_type, json=True)

    def check_create_delete(self, image_type):
        """dataset - create, delete"""
        job_id = self.create_quick_dataset(image_type)
        assert self.delete_dataset(job_id) == 200, 'delete failed'
        assert not self.dataset_exists(job_id), 'dataset exists after delete'

    def check_create_wait_delete(self, image_type):
        """dataset - create, wait, delete"""
        job_id = self.create_quick_dataset(image_type)
        assert self.dataset_wait_completion(job_id) == 'Done', 'create failed'
        assert self.delete_dataset(job_id) == 200, 'delete failed'
        assert not self.dataset_exists(job_id), 'dataset exists after delete'

    def check_create_abort_delete(self, image_type):
        """dataset - create, abort, delete"""
        job_id = self.create_quick_dataset(image_type)
        assert self.abort_dataset(job_id) == 200, 'abort failed'
        assert self.delete_dataset(job_id) == 200, 'delete failed'
        assert not self.dataset_exists(job_id), 'dataset exists after delete'

    def check_textfile(self, image_type, absolute_path):
        """any image type, and absolute or relative path"""
        job_id = self.create_from_textfiles(image_type, absolute_path=absolute_path)
        assert self.dataset_wait_completion(job_id) == 'Done', 'create failed'

    def create_from_textfiles(self, image_type, absolute_path=True):
        """
        Create a dataset from textfiles

        Arguments:
        absolute_path -- if False, give relative paths and image folders
        """
        textfile_train_images = ''
        textfile_labels_file = ''
        label_id = 0
        for (label, images) in self.image_type_data[image_type].images.iteritems():
            textfile_labels_file += '%s\n' % label
            for image in images:
                image_path = image
                if absolute_path:
                    image_path = os.path.join(self.image_type_data[image_type].data_path, image_path)
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
            data['textfile_train_folder'] = self.image_type_data[image_type].data_path
            data['textfile_val_folder'] = self.image_type_data[image_type].data_path

        return self.create_dataset(**data)

    def check_nonsquare_dimensions(self, image_type):
        """dataset - nonsquare dimensions"""
        job_id = self.create_quick_dataset(
                image_type,
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
        cls.datasets = {image_type: cls.create_quick_dataset(image_type) for image_type in ImageType.TYPES}
        for dataset_id in cls.datasets.itervalues():
            assert cls.dataset_wait_completion(dataset_id) == 'Done', 'dataset creation failed'

    def test_alltests(self):
        for image_type in ImageType.TYPES:
            yield self.check_index_json, image_type
            yield self.check_dataset_json, image_type

    def check_index_json(self, image_type):
        """created dataset - index.json"""
        rv = self.app.get('/index.json')
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        found = False
        for d in content['datasets']:
            if d['id'] == self.datasets[image_type]:
                found = True
                break
        assert found, 'dataset not found in list'

    def check_dataset_json(self, image_type):
        """created dataset - json"""
        rv = self.app.get('/datasets/%s.json' % self.datasets[image_type])
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        assert content['id'] == self.datasets[image_type], 'expected different job_id'

class TestModelCreation(WebappBaseTest):
    """
    Model creation tests
    """
    @classmethod
    def setUpClass(cls):
        super(TestModelCreation, cls).setUpClass()
        cls.datasets = {image_type: cls.create_dataset(
                method = 'folder',
                folder_train = dataset.data_path,
                resize_width = DUMMY_IMAGE_DIM,
                resize_height = DUMMY_IMAGE_DIM,
                ) for image_type, dataset in cls.image_type_data.iteritems()}

    def test_page_model_new(self):
        """new image classification model page"""
        rv = self.app.get('/models/images/classification/new')
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        assert 'New Image Classification Model' in rv.data, 'unexpected page format'

    def test_visualize_network(self):
        """visualize network"""
        rv = self.app.post('/models/visualize-network',
                data = {'custom_network': dummy_network}
                )
        s = BeautifulSoup(rv.data)
        body = s.select('body')
        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)
        image = s.select('img')
        assert image is not None, "didn't return an image"

    def test_alltests(self):
        for image_type in ImageType.TYPES:
            yield self.check_create_json, image_type
            yield self.check_create_delete, image_type
            yield self.check_create_wait_delete, image_type
            yield self.check_create_abort_delete, image_type
            yield self.check_snapshot_interval_2, image_type
            yield self.check_snapshot_interval_0_5, image_type



    def check_create_json(self, image_type):
        """model - create w/ json"""
        self.create_quick_model(self.datasets[image_type], json=True)

    def check_create_delete(self, image_type):
        """model - create, delete"""
        job_id = self.create_quick_model(self.datasets[image_type])
        assert self.delete_model(job_id) == 200, 'delete failed'
        assert not self.model_exists(job_id), 'model exists after delete'

    def check_create_wait_delete(self, image_type):
        """model - create, wait, delete"""
        job_id = self.create_quick_model(self.datasets[image_type])
        assert self.model_wait_completion(job_id) == 'Done', 'create failed'
        assert self.delete_model(job_id) == 200, 'delete failed'
        assert not self.model_exists(job_id), 'model exists after delete'

    def check_create_abort_delete(self, image_type):
        """model - create, abort, delete"""
        job_id = self.create_quick_model(self.datasets[image_type])
        assert self.abort_model(job_id) == 200, 'abort failed'
        assert self.delete_model(job_id) == 200, 'delete failed'
        assert not self.model_exists(job_id), 'model exists after delete'

    def check_snapshot_interval_2(self, image_type):
        """model - snapshot_interval 2"""
        job_id = self.create_quick_model(self.datasets[image_type], train_epochs=1, snapshot_interval=0.5)
        assert self.model_wait_completion(job_id) == 'Done', 'create failed'
        rv = self.app.get('/models/%s.json' % job_id)
        assert rv.status_code == 200, 'json load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        assert len(content['snapshots']) > 1, 'should take >1 snapshot'

    def check_snapshot_interval_0_5(self, image_type):
        """model - snapshot_interval 0.5"""
        job_id = self.create_quick_model(self.datasets[image_type], train_epochs=4, snapshot_interval=2)
        assert self.model_wait_completion(job_id) == 'Done', 'create failed'
        rv = self.app.get('/models/%s.json' % job_id)
        assert rv.status_code == 200, 'json load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        assert len(content['snapshots']) == 2, 'should take 2 snapshots'

    # for the GPU tests, only test the first dataset.

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
        job_id = self.create_quick_model(self.datasets[ImageType.COLOR], select_gpu=gpu_index)
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
        job_id = self.create_quick_model(self.datasets[ImageType.COLOR],
                select_gpus_list=','.join(gpu_list))
        assert self.delete_model(job_id) == 200, 'delete failed'

class TestCreatedModel(WebappBaseTest):
    """
    Tests on a model that has already been created
    """
    @classmethod
    def setUpClass(cls):
        super(TestCreatedModel, cls).setUpClass()
        datasets = [(cls.create_quick_dataset(image_type), image_type) for image_type in ImageType.TYPES]
        cls.id_map = {}

        class Ids(object):
            def __init__(self, model_id, dataset_id):
                self.model_id, self.dataset_id = model_id, dataset_id

        for dataset_id, image_type in datasets:
            assert cls.dataset_wait_completion(dataset_id) == 'Done', 'dataset creation failed'
            model_id = cls.create_quick_model(dataset_id)
            assert cls.model_wait_completion(model_id) == 'Done', 'model creation failed'
            cls.id_map[image_type] = Ids(model_id, dataset_id)

    def test_alltests(self):
        for (image_type, ids) in self.id_map.iteritems():
            yield self.check_save, image_type
            # check_download yields on its own
            self.check_download(image_type)
            yield self.check_index_json, image_type
            yield self.check_model_json, image_type
            yield self.check_classify_one, image_type
            yield self.check_classify_one_json, image_type
            yield self.check_classify_many, image_type
            yield self.check_classify_many_json, image_type
            yield self.check_top_n, image_type
            yield self.check_retrain, image_type

    def check_save(self, image_type):
        """created model - save"""
        job = webapp.scheduler.get_job(self.id_map[image_type].model_id)
        assert job.save(), 'Job failed to save'

    def download_model(self, image_type, extension):
        url = '/models/%s/download.%s' % (self.id_map[image_type].model_id, extension)
        rv = self.app.get(url)
        assert rv.status_code == 200, 'download "%s" failed with %s' % (url, rv.status_code)

    def check_download(self, image_type):
        """created model - download"""
        for extension in ['tar', 'zip', 'tar.gz', 'tar.bz2']:
            yield self.download_model, self.id_map[image_type].model_id, extension

    def check_index_json(self, image_type):
        """created model - index.json"""
        rv = self.app.get('/index.json')
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        found = False
        for m in content['models']:
            if m['id'] == self.id_map[image_type].model_id:
                found = True
                break
        assert found, 'model not found in list'

    def check_model_json(self, image_type):
        """created model - json"""
        rv = self.app.get('/models/%s.json' % self.id_map[image_type].model_id)
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        assert content['id'] == self.id_map[image_type].model_id, 'expected different job_id'
        assert len(content['snapshots']) > 0, 'no snapshots in list'

    def check_classify_one(self, image_type):
        """created model - classify one"""
        category = next(iter(LABELS))
        image_path = self.image_type_data[image_type].images[category][0]
        image_path = os.path.join(self.image_type_data[image_type].data_path, image_path)
        with open(image_path) as infile:
            # StringIO wrapping is needed to simulate POST file upload.
            image_upload = (StringIO(infile.read()), 'image.png')

        rv = self.app.post(
                '/models/images/classification/classify_one?job_id=%s' % self.id_map[image_type].model_id,
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

    def check_classify_one_json(self, image_type):
        """created model - classify one JSON"""
        category = next(iter(LABELS))
        image_path = self.image_type_data[image_type].images[category][0]
        image_path = os.path.join(self.image_type_data[image_type].data_path, image_path)
        with open(image_path) as infile:
            # StringIO wrapping is needed to simulate POST file upload.
            image_upload = (StringIO(infile.read()), 'image.png')

        rv = self.app.post(
                '/models/images/classification/classify_one.json?job_id=%s' % self.id_map[image_type].model_id,
                data = {
                    'image_file': image_upload,
                    'show_visualizations': 'y',
                    }
                )
        assert rv.status_code == 200, 'POST failed with %s' % rv.status_code
        data = json.loads(rv.data)
        assert data['predictions'][0][0] == category, 'image misclassified'

    def check_classify_many(self, image_type):
        """created model - classify many"""
        textfile_images = ''
        label_id = 0
        for (label, images) in self.image_type_data[image_type].images.iteritems():
            for image in images:
                image_path = image
                image_path = os.path.join(self.image_type_data[image_type].data_path, image_path)
                textfile_images += '%s %d\n' % (image_path, label_id)
            label_id += 1

        # StringIO wrapping is needed to simulate POST file upload.
        file_upload = (StringIO(textfile_images), 'images.txt')

        rv = self.app.post(
                '/models/images/classification/classify_many?job_id=%s' % self.id_map[image_type].model_id,
                data = {'image_list': file_upload}
                )
        s = BeautifulSoup(rv.data)
        body = s.select('body')
        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)

    def check_classify_many_json(self, image_type):
        """created model - classify many JSON"""
        textfile_images = ''
        label_id = 0
        for (label, images) in self.image_type_data[image_type].images.iteritems():
            for image in images:
                image_path = image
                image_path = os.path.join(self.image_type_data[image_type].data_path, image_path)
                textfile_images += '%s %d\n' % (image_path, label_id)
            label_id += 1

        # StringIO wrapping is needed to simulate POST file upload.
        file_upload = (StringIO(textfile_images), 'images.txt')

        rv = self.app.post(
                '/models/images/classification/classify_many.json?job_id=%s' % self.id_map[image_type].model_id,
                data = {'image_list': file_upload}
                )
        assert rv.status_code == 200, 'POST failed with %s' % rv.status_code
        data = json.loads(rv.data)
        assert 'classifications' in data, 'invalid response'

    def check_top_n(self, image_type):
        """created model - top n predictions"""
        textfile_images = ''
        label_id = 0
        for (label, images) in self.image_type_data[image_type].images.iteritems():
            for image in images:
                image_path = image
                image_path = os.path.join(self.image_type_data[image_type].data_path, image_path)
                textfile_images += '%s %d\n' % (image_path, label_id)
            label_id += 1

        # StringIO wrapping is needed to simulate POST file upload.
        file_upload = (StringIO(textfile_images), 'images.txt')

        rv = self.app.post(
                '/models/images/classification/top_n?job_id=%s' % self.id_map[image_type].model_id,
                data = {'image_list': file_upload}
                )
        s = BeautifulSoup(rv.data)
        body = s.select('body')
        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)
        keys = self.image_type_data[image_type].images.keys()
        for key in keys:
            assert key in rv.data, '"%s" not found in the response'

    def check_retrain(self, image_type):
        """created model - retrain"""
        options = {}
        options['previous_networks'] = self.id_map[image_type].model_id
        rv = self.app.get('/models/%s.json' % self.id_map[image_type].model_id)
        assert rv.status_code == 200, 'json load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        assert len(content['snapshots']), 'should have at least snapshot'
        options['%s-snapshot' % self.id_map[image_type].model_id] = content['snapshots'][-1]
        job_id = self.create_quick_model(self.id_map[image_type].dataset_id,
                method='previous', **options)
        self.abort_model(job_id)

class TestDatasetModelInteractions(WebappBaseTest):
    """
    Test the interactions between datasets and models
    """
    def test_alltests(self):
        for image_type in ImageType.TYPES:
            yield self.check_create_model_deleted_dataset, image_type
            yield self.check_create_model_running_dataset, image_type
            yield self.check_delete_dataset_dependent_model, image_type
            yield self.check_delete_running_dataset_dependent_model, image_type

    def check_create_model_deleted_dataset(self, image_type):
        """
        If you try to create a model using a deleted dataset, it should fail
        """
        dataset_id = self.create_quick_dataset(image_type)
        assert self.delete_dataset(dataset_id) == 200, 'delete failed'
        assert not self.dataset_exists(dataset_id), 'dataset exists after delete'

        try:
            model_id = self.create_quick_model(dataset_id)
        except RuntimeError:
            return
        assert False, 'Should have failed'

    def check_create_model_running_dataset(self, image_type):
        """
        If you try to create a model using a running dataset,
        it should wait to start until the dataset is completed
        """
        dataset_id = self.create_quick_dataset(image_type)
        model_id = self.create_quick_model(dataset_id)

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

    def check_delete_dataset_dependent_model(self, image_type):
        """
        If you try to delete a completed dataset with a dependent model, it should fail
        """
        dataset_id = self.create_quick_dataset(image_type)
        model_id = self.create_quick_model(dataset_id)
        assert self.dataset_wait_completion(dataset_id) == 'Done', 'dataset creation failed'
        assert self.delete_dataset(dataset_id) == 403, 'dataset deletion should not have succeeded'
        self.abort_model(model_id)

    def check_delete_running_dataset_dependent_model(self, image_type):
        """
        If you try to delete a running dataset with a dependent model, it should fail
        """
        dataset_id = self.create_quick_dataset(image_type)
        model_id = self.create_quick_model(dataset_id)
        assert self.delete_dataset(dataset_id) == 403, 'dataset deletion should not have succeeded'
        self.abort_dataset(dataset_id)
        self.abort_model(model_id)

