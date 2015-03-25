# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import json
import os
import shutil
import tempfile
import time
import unittest

from gevent import monkey
monkey.patch_all()

import numpy as np
from skimage import io
from urlparse import urlparse
from cStringIO import StringIO

import webapp


def get_dummy_network():
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


def create_rgb_dataset(data_path):
    dim = 64
    count = 10
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

            img_path = os.path.join(label_path, str(i) + '.png')
            io.imsave(os.path.join(data_path, img_path), img)
            images[label].append(img_path)

    return images


class BaseTestCase(unittest.TestCase):

    @classmethod
    def setup_class(cls):
        cls.data_path = tempfile.mkdtemp()
        cls.images = create_rgb_dataset(cls.data_path)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.data_path)

    def setUp(self):
        webapp.scheduler.start()
        webapp.app.config['WTF_CSRF_ENABLED'] = False
        webapp.app.config['debug'] = True
        self.app = webapp.app.test_client()
        self.server = 'http://0.0.0.0:5000/'
        self.jobs = self.server + '/jobs/'
        self.created_jobs = []

    def tearDown(self):
        # If a test fail, some jobs might not be deleted correctly, try to cleanup all created jobs here.
        for job in self.created_jobs:
            self.job_try_delete(job)

        # Do not stop the scheduler here, since this action is
        # asynchronous. This would likely cause the next test to fail.

    def job_exists(self, job_name):
        job_url = self.jobs + job_name
        rv = self.app.get(job_url, follow_redirects=True)
        rv.close()
        assert rv.status_code == 200 or rv.status_code == 404
        return rv.status_code == 200

    def extract_name(self, rv):
        job_url = rv.headers['Location']
        parsed_url = urlparse(job_url)
        job_name = parsed_url.path.split('/')[-1]
        return job_name

    def dataset_create_folder(self, name, folder):
        create_url = self.server + '/datasets/images/classification'
        body = {'dataset_name': name, 'method': 'folder', 'folder_train': folder}
        rv = self.app.post(create_url, data=body)
        rv.close()
        assert rv.status_code >= 300 and rv.status_code <= 310, 'No redirect after dataset creation'

        job_name = self.extract_name(rv)
        assert self.job_exists(job_name)

        self.created_jobs.append(job_name)
        return job_name

    def dataset_create_textfile(self, name, absolute_path=True):
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
        body = {'dataset_name': name, 'method': 'textfile', 'textfile_train_images': train_upload,
                'textfile_use_val': 'y', 'textfile_val_images': val_upload,
                'textfile_labels_file': labels_upload}
        if not absolute_path:
            body['textfile_train_folder'] = self.data_path
            body['textfile_val_folder'] = self.data_path

        create_url = self.server + '/datasets/images/classification'
        rv = self.app.post(create_url, data=body)
        assert rv.status_code >= 300 and rv.status_code <= 310, 'No redirect after dataset creation'

        job_name = self.extract_name(rv)
        assert self.job_exists(job_name)

        self.created_jobs.append(job_name)
        return job_name


    def model_create(self, name, dataset):
        network = get_dummy_network()
        create_url = self.server + '/models/images/classification'
        body = {'model_name': name, 'dataset': dataset, 'method': 'custom', 'custom_network': network}
        rv = self.app.post(create_url, data=body)
        rv.close()
        assert rv.status_code >= 300 and rv.status_code <= 310, 'No redirect after model creation'

        job_name = self.extract_name(rv)
        assert self.job_exists(job_name)

        self.created_jobs.append(job_name)
        return job_name

    def model_download(self, job_name, epoch):
        body = {'snapshot_epoch': epoch}
        download_url = self.server + '/models/' + job_name + '/download_snapshot'
        rv = self.app.post(download_url, data=body)
        rv.close()
        assert rv.status_code == 200

    def job_status(self, job_name):
        status_url = self.jobs + job_name + '/status'
        rv = self.app.get(status_url)
        assert rv.status_code == 200, 'Cannot get status of job %s' % job_name
        status = json.loads(rv.data)
        return status

    def job_wait_completion(self, job_name, timeout, polling_period=0.5):
        elapsed = 0
        while True:
            status = self.job_status(job_name)
            if status['status'] == 'Done':
                break
            assert status['status'] in ['Initialized', 'Waiting', 'Running'], 'Invalid job status: %s' % status['status']
            time.sleep(polling_period)
            elapsed += polling_period
            assert elapsed < timeout, 'Job completion timeout'

    def job_abort(self, job_name):
        abort_url = self.jobs + job_name + '/abort'
        self.app.post(abort_url)

    def job_try_delete(self, job_name):
        rv = self.app.delete('/jobs/' + job_name)
        rv.close()

    def job_delete(self, job_name):
        rv = self.app.delete('/jobs/' + job_name)
        assert rv.status_code == 200
        rv.close()

    def job_delete_code(self, job_name):
        rv = self.app.delete('/jobs/' + job_name)
        rc = rv.status_code
        rv.close()
        return rc


class WebappTestCase(BaseTestCase):

    def test_page_home(self):
        rv = self.app.get('/')
        assert rv.status_code == 200, 'code is %s' % rv.status_code
        for h in ['Home', 'Datasets', 'Models']:
            assert h in rv.data

    def test_page_dataset_new(self):
        dataset_new_url = self.server + '/datasets/images/classification/new'
        rv = self.app.get(dataset_new_url)
        assert rv.status_code == 200
        assert 'New Image Classification Dataset' in rv.data

    def test_page_model_new(self):
        model_new_url = self.server + '/models/images/classification/new'
        rv = self.app.get(model_new_url)
        assert rv.status_code == 200
        assert 'New Image Classification Model' in rv.data

    def test_invalid_page(self):
        rv = self.app.get('/foo')
        assert rv.status_code == 404

    def test_invalid_job(self):
        assert not self.job_exists('foo'), 'Invalid job query should return 404'

    def test_dataset_create_delete(self):
        dataset_name = self.dataset_create_folder('rgb_dataset', self.data_path)
        self.job_delete(dataset_name)
        assert not self.job_exists(dataset_name), 'Job was not deleted'

    def test_dataset_create_invalid(self):
        empty_dir = tempfile.mkdtemp()
        dataset_name = self.dataset_create_folder('rgb_dataset', empty_dir)
        time.sleep(3)
        status = self.job_status(dataset_name)
        assert status['status'] == 'Error'

    def test_dataset_create_wait_delete(self):
        dataset_name = self.dataset_create_folder('rgb_dataset', self.data_path)
        self.job_wait_completion(dataset_name, 10)
        self.job_delete(dataset_name)
        assert not self.job_exists(dataset_name), 'Job was not deleted'

    def test_dataset_create_abort_delete(self):
        dataset_name = self.dataset_create_folder('rgb_dataset', self.data_path)
        self.job_abort(dataset_name)
        self.job_delete(dataset_name)
        assert not self.job_exists(dataset_name), 'Job was not deleted'

    def test_model_create_delete(self):
        dataset_name = self.dataset_create_folder('rgb_dataset', self.data_path)
        model_name = self.model_create('rgb_model', dataset_name)
        self.job_delete(model_name)
        self.job_delete(dataset_name)
        assert not self.job_exists(dataset_name), 'Job was not deleted'
        assert not self.job_exists(model_name), 'Job was not deleted'

    # Concurrently delete a dataset and create a model.
    def test_model_create_with_deleted_database(self):
        dataset_name = self.dataset_create_folder('rgb_dataset', self.data_path)
        self.job_wait_completion(dataset_name, 10)
        self.job_delete(dataset_name)
        try:
            model_name = self.model_create('rgb_model', dataset_name)
        except AssertionError:
            pass
        else:
            self.job_delete(model_name)
            assert False, 'Model creation should have failed'

    def test_model_wait_create_delete(self):
        dataset_name = self.dataset_create_folder('rgb_dataset', self.data_path)
        self.job_wait_completion(dataset_name, 10)
        model_name = self.model_create('rgb_model', dataset_name)
        self.job_delete(model_name)
        self.job_delete(dataset_name)
        assert not self.job_exists(model_name), 'ModelJob was not deleted'
        assert not self.job_exists(dataset_name), 'DatasetJob was not deleted'

    def test_model_wait_create_wait_delete(self):
        dataset_name = self.dataset_create_folder('rgb_dataset', self.data_path)
        self.job_wait_completion(dataset_name, 10)
        model_name = self.model_create('rgb_model', dataset_name)
        self.job_wait_completion(model_name, 30)
        self.job_delete(model_name)
        self.job_delete(dataset_name)
        assert not self.job_exists(model_name), 'ModelJob was not deleted'
        assert not self.job_exists(dataset_name), 'DatasetJob was not deleted'

    def test_model_download(self):
        dataset_name = self.dataset_create_folder('rgb_dataset', self.data_path)
        self.job_wait_completion(dataset_name, 10)
        model_name = self.model_create('rgb_model', dataset_name)
        self.job_wait_completion(model_name, 30)

        self.model_download(model_name, 1)

        self.job_delete(model_name)
        self.job_delete(dataset_name)
        assert not self.job_exists(model_name), 'ModelJob was not deleted'
        assert not self.job_exists(dataset_name), 'DatasetJob was not deleted'

    def test_model_create_wait_delete(self):
        """
        Create model while dataset still running
        """
        dataset_name = self.dataset_create_folder('rgb_dataset', self.data_path)
        model_name = self.model_create('rgb_model', dataset_name)
        self.job_wait_completion(model_name, 10)
        self.job_delete(model_name)
        self.job_delete(dataset_name)
        assert not self.job_exists(model_name), 'ModelJob was not deleted'
        assert not self.job_exists(dataset_name), 'DatasetJob was not deleted'

    # A dataset should not be deleted while a model using it is running.
    def test_model_create_dataset_delete(self):
        """
        Delete dataset while model still running
        """
        dataset_name = self.dataset_create_folder('rgb_dataset', self.data_path)
        model_name = self.model_create('rgb_model', dataset_name)
        assert self.job_delete_code(dataset_name) == 403, 'Job should not have been deleted'
        self.job_delete(model_name)
        self.job_delete(dataset_name)
        assert not self.job_exists(model_name), 'ModelJob was not deleted'
        assert not self.job_exists(dataset_name), 'DatasetJob was not deleted'

    def test_textfile_absolute_path(self):
        dataset_name = self.dataset_create_textfile('rgb_dataset')
        self.job_wait_completion(dataset_name, 10)
        model_name = self.model_create('rgb_model', dataset_name)
        self.job_wait_completion(model_name, 30)
        self.job_delete(model_name)
        self.job_delete(dataset_name)
        assert not self.job_exists(model_name), 'ModelJob was not deleted'
        assert not self.job_exists(dataset_name), 'DatasetJob was not deleted'

    def test_textfile_relative_path(self):
        dataset_name = self.dataset_create_textfile('rgb_dataset', absolute_path=False)
        self.job_wait_completion(dataset_name, 10)
        model_name = self.model_create('rgb_model', dataset_name)
        self.job_wait_completion(model_name, 30)
        self.job_delete(model_name)
        self.job_delete(dataset_name)
        assert not self.job_exists(model_name), 'ModelJob was not deleted'
        assert not self.job_exists(dataset_name), 'DatasetJob was not deleted'
