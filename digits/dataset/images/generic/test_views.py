# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import json
import os
import tempfile

from bs4 import BeautifulSoup

from .test_lmdb_creator import create_lmdbs
from digits import test_utils
import digits.test_views

# May be too short on a slow system
TIMEOUT_DATASET = 45

################################################################################
# Base classes (they don't start with "Test" so nose won't run them)
################################################################################


class BaseViewsTest(digits.test_views.BaseViewsTest):
    """
    Provides some functions
    """
    @classmethod
    def dataset_exists(cls, job_id):
        return cls.job_exists(job_id, 'datasets')

    @classmethod
    def dataset_status(cls, job_id):
        return cls.job_status(job_id, 'datasets')

    @classmethod
    def abort_dataset(cls, job_id):
        return cls.abort_job(job_id, job_type='datasets')

    @classmethod
    def dataset_wait_completion(cls, job_id, **kwargs):
        kwargs['job_type'] = 'datasets'
        if 'timeout' not in kwargs:
            kwargs['timeout'] = TIMEOUT_DATASET
        return cls.job_wait_completion(job_id, **kwargs)

    @classmethod
    def delete_dataset(cls, job_id):
        return cls.delete_job(job_id, job_type='datasets')


class BaseViewsTestWithImageset(BaseViewsTest):
    """
    Provides some LMDBs and some functions
    """

    @classmethod
    def setUpClass(cls):
        super(BaseViewsTestWithImageset, cls).setUpClass()
        if not hasattr(BaseViewsTestWithImageset, 'imageset_folder'):
            # Create folder and LMDBs for all test classes
            BaseViewsTestWithImageset.imageset_folder = tempfile.mkdtemp()
            BaseViewsTestWithImageset.test_image = create_lmdbs(BaseViewsTestWithImageset.imageset_folder)
            BaseViewsTestWithImageset.val_db_path = os.path.join(
                BaseViewsTestWithImageset.imageset_folder,
                'val_images')
        cls.created_datasets = []

    @classmethod
    def tearDownClass(cls):
        # delete any created datasets
        for job_id in cls.created_datasets:
            cls.delete_dataset(job_id)
        super(BaseViewsTestWithImageset, cls).tearDownClass()

    @classmethod
    def create_dataset(cls, **kwargs):
        """
        Create a dataset
        Returns the job_id
        Raises RuntimeError if job fails to create

        Keyword arguments:
        **kwargs -- data to be sent with POST request
        """
        data = {
            'dataset_name':     'test_dataset',
            'group_name':       'test_group',
            'method':           'prebuilt',
            'prebuilt_train_images': os.path.join(cls.imageset_folder, 'train_images'),
            'prebuilt_train_labels': os.path.join(cls.imageset_folder, 'train_labels'),
            'prebuilt_val_images': os.path.join(cls.imageset_folder, 'val_images'),
            'prebuilt_val_labels': os.path.join(cls.imageset_folder, 'val_labels'),
            'prebuilt_mean_file': os.path.join(cls.imageset_folder, 'train_mean.binaryproto'),
        }
        data.update(kwargs)

        request_json = data.pop('json', False)
        url = '/datasets/images/generic'
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
            s = BeautifulSoup(rv.data, 'html.parser')
            div = s.select('div.alert-danger')
            if div:
                print div[0]
            else:
                print rv.data
            raise RuntimeError('Failed to create dataset - status %s' % rv.status_code)

        job_id = cls.job_id_from_response(rv)

        assert cls.dataset_exists(job_id), 'dataset not found after successful creation'

        cls.created_datasets.append(job_id)
        return job_id


class BaseViewsTestWithDataset(BaseViewsTestWithImageset):
    """
    Provides a dataset and some functions
    """
    @classmethod
    def setUpClass(cls):
        super(BaseViewsTestWithDataset, cls).setUpClass()
        cls.dataset_id = cls.create_dataset(json=True)
        assert cls.dataset_wait_completion(cls.dataset_id) == 'Done', 'create failed'

################################################################################
# Test classes
################################################################################


class TestViews(BaseViewsTest, test_utils.DatasetMixin):
    """
    Tests which don't require an imageset or a dataset
    """

    def test_page_dataset_new(self):
        rv = self.app.get('/datasets/images/generic/new')
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        assert 'New Image Dataset' in rv.data, 'unexpected page format'

    def test_nonexistent_dataset(self):
        assert not self.dataset_exists('foo'), "dataset shouldn't exist"


class TestCreation(BaseViewsTestWithImageset, test_utils.DatasetMixin):
    """
    Dataset creation tests
    """

    def test_bad_path(self):
        try:
            self.create_dataset(
                prebuilt_train_images='/not-a-directory'
            )
        except RuntimeError:
            return
        raise AssertionError('Should have failed')

    def test_create_json(self):
        job_id = self.create_dataset(json=True)
        self.abort_dataset(job_id)

    def test_create_delete(self):
        job_id = self.create_dataset()
        assert self.delete_dataset(job_id) == 200, 'delete failed'
        assert not self.dataset_exists(job_id), 'dataset exists after delete'

    def test_create_abort_delete(self):
        job_id = self.create_dataset()
        assert self.abort_dataset(job_id) == 200, 'abort failed'
        assert self.delete_dataset(job_id) == 200, 'delete failed'
        assert not self.dataset_exists(job_id), 'dataset exists after delete'

    def test_create_wait_delete(self):
        job_id = self.create_dataset()
        assert self.dataset_wait_completion(job_id) == 'Done', 'create failed'
        assert self.delete_dataset(job_id) == 200, 'delete failed'
        assert not self.dataset_exists(job_id), 'dataset exists after delete'

    def test_no_force_same_shape(self):
        job_id = self.create_dataset(force_same_shape=0)
        assert self.dataset_wait_completion(job_id) == 'Done', 'create failed'

    def test_clone(self):
        options_1 = {
            'resize_channels': '1',
        }

        job1_id = self.create_dataset(**options_1)
        assert self.dataset_wait_completion(job1_id) == 'Done', 'first job failed'
        rv = self.app.get('/datasets/%s.json' % job1_id)
        assert rv.status_code == 200, 'json load failed with %s' % rv.status_code
        content1 = json.loads(rv.data)

        # Clone job1 as job2
        options_2 = {
            'clone': job1_id,
        }

        job2_id = self.create_dataset(**options_2)
        assert self.dataset_wait_completion(job2_id) == 'Done', 'second job failed'
        rv = self.app.get('/datasets/%s.json' % job2_id)
        assert rv.status_code == 200, 'json load failed with %s' % rv.status_code
        content2 = json.loads(rv.data)

        # These will be different
        content1.pop('id')
        content2.pop('id')
        content1.pop('directory')
        content2.pop('directory')
        assert (content1 == content2), 'job content does not match'

        job1 = digits.webapp.scheduler.get_job(job1_id)
        job2 = digits.webapp.scheduler.get_job(job2_id)

        assert (job1.form_data == job2.form_data), 'form content does not match'


class TestCreated(BaseViewsTestWithDataset, test_utils.DatasetMixin):
    """
    Tests on a dataset that has already been created
    """

    def test_index_json(self):
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
        rv = self.app.get('/datasets/%s.json' % self.dataset_id)
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        content = json.loads(rv.data)
        assert content['id'] == self.dataset_id, 'expected different job_id'

    def test_edit_name(self):
        status = self.edit_job(
            self.dataset_id,
            name='new name'
        )
        assert status == 200, 'failed with %s' % status
        rv = self.app.get('/datasets/summary?job_id=%s' % self.dataset_id)
        assert rv.status_code == 200
        assert 'new name' in rv.data

    def test_edit_notes(self):
        status = self.edit_job(
            self.dataset_id,
            notes='new notes'
        )
        assert status == 200, 'failed with %s' % status
