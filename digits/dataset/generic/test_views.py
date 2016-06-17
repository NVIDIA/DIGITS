# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import json
import os

from bs4 import BeautifulSoup

import digits.test_views
from digits import extensions
from digits.utils import constants

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


class BaseViewsTestWithDataset(BaseViewsTest):
    """
    Provides some functions
    """

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
            'dataset_name': 'test_dataset',
            }
        data.update(kwargs)

        request_json = data.pop('json', False)
        url = '/datasets/generic/create/%s' % cls.EXTENSION_ID
        if request_json:
            url += '.json'

        rv = cls.app.post(url, data=data)

        if request_json:
            if rv.status_code != 200:
                raise RuntimeError(
                    'Model creation failed with %s' % rv.status_code)
            return json.loads(rv.data)['id']

        # expect a redirect
        if not 300 <= rv.status_code <= 310:
            s = BeautifulSoup(rv.data, 'html.parser')
            div = s.select('div.alert-danger')
            if div:
                print div[0]
            else:
                print rv.data
            raise RuntimeError(
                'Failed to create dataset - status %s' % rv.status_code)

        job_id = cls.job_id_from_response(rv)

        assert cls.dataset_exists(job_id), 'dataset not found after successful creation'

        cls.created_datasets.append(job_id)
        return job_id

    @classmethod
    def get_dataset_json(cls):
        rv = cls.app.get('/datasets/%s.json' % cls.dataset_id)
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        return json.loads(rv.data)

    @classmethod
    def setUpClass(cls, **kwargs):
        super(BaseViewsTestWithDataset, cls).setUpClass()
        cls.dataset_id = cls.create_dataset(json=True, **kwargs)
        assert cls.dataset_wait_completion(cls.dataset_id) == 'Done', 'create failed'
        # Save val DB path
        json = cls.get_dataset_json()
        for t in json['create_db_tasks']:
            if t['stage'] == constants.VAL_DB:
                if t['feature_db_path'] is not None:
                    cls.val_db_path = os.path.join(
                        json['directory'],
                        t['feature_db_path'])
                else:
                    cls.val_db_path = None


class GenericViewsTest(BaseViewsTest):
    def test_page_dataset_new(self):
        rv = self.app.get('/datasets/generic/new/%s' % self.EXTENSION_ID)
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        assert extensions.data.get_extension(self.EXTENSION_ID).get_title() in rv.data, 'unexpected page format'

    def test_nonexistent_dataset(self):
        assert not self.dataset_exists('foo'), "dataset shouldn't exist"


class GenericCreationTest(BaseViewsTestWithDataset):
    """
    Dataset creation tests
    """

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

    def test_invalid_number_of_reader_threads(self):
        try:
            self.create_dataset(
                json=True,
                dsopts_num_threads=0)
            assert False
        except RuntimeError:
            # job is expected to fail with a RuntimeError
            pass

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

        ## Clone job1 as job2
        options_2 = {
            'clone': job1_id,
        }

        job2_id = self.create_dataset(**options_2)
        assert self.dataset_wait_completion(job2_id) == 'Done', 'second job failed'
        rv = self.app.get('/datasets/%s.json' % job2_id)
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


class GenericCreatedTest(BaseViewsTestWithDataset):
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
        content = self.get_dataset_json()
        assert content['id'] == self.dataset_id, 'expected same job_id: %s != %s' % (content['id'], self.dataset_id)

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

    def test_explore_features(self):
        # features DB is encoded by default
        rv = self.app.get('/datasets/generic/explore?db=train_db%%2Ffeatures&job_id=%s' % self.dataset_id)
        # just make sure this doesn't return an error
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code

################################################################################
# Test classes
################################################################################


class TestImageGradientViews(GenericViewsTest):
    """
    Tests which don't require an imageset or a dataset
    """
    EXTENSION_ID = "image-gradients"


class TestImageGradientCreation(GenericCreationTest):
    """
    Test that create datasets
    """
    EXTENSION_ID = "image-gradients"

    @classmethod
    def setUpClass(cls, **kwargs):
        super(TestImageGradientCreation, cls).setUpClass(train_image_count=100)


class TestImageGradientCreated(GenericCreatedTest):
    """
    Test that create datasets
    """
    EXTENSION_ID = "image-gradients"

    @classmethod
    def setUpClass(cls, **kwargs):
        super(TestImageGradientCreated, cls).setUpClass(train_image_count=100)
