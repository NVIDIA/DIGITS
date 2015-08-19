# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import json
import os
import shutil
import tempfile
import time
import unittest
import itertools
import urllib

from gevent import monkey
monkey.patch_all()
from bs4 import BeautifulSoup
import PIL.Image
from urlparse import urlparse
from cStringIO import StringIO

import digits.test_views
from test_imageset_creator import create_classification_imageset, IMAGE_SIZE as DUMMY_IMAGE_SIZE, IMAGE_COUNT as DUMMY_IMAGE_COUNT

# May be too short on a slow system
TIMEOUT_DATASET = 15

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
    def dataset_info(cls, job_id):
        return cls.job_info(job_id, 'datasets')

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
    Provides an imageset and some functions
    """
    # Inherited classes may want to override these attributes
    IMAGE_HEIGHT    = 10
    IMAGE_WIDTH     = 10
    IMAGE_CHANNELS  = 3

    UNBALANCED_CATEGORY = False

    @classmethod
    def setUpClass(cls):
        super(BaseViewsTestWithImageset, cls).setUpClass()
        cls.imageset_folder = tempfile.mkdtemp()
        # create imageset
        cls.imageset_paths = create_classification_imageset(cls.imageset_folder,
                                                            add_unbalanced_category=cls.UNBALANCED_CATEGORY)
        cls.created_datasets = []

    @classmethod
    def tearDownClass(cls):
        # delete any created datasets
        for job_id in cls.created_datasets:
            cls.delete_dataset(job_id)
        # delete imageset
        shutil.rmtree(cls.imageset_folder)
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
                'method':           'folder',
                'folder_train':     cls.imageset_folder,
                'resize_channels':  cls.IMAGE_CHANNELS,
                'resize_width':     cls.IMAGE_WIDTH,
                'resize_height':    cls.IMAGE_HEIGHT,
                }
        data.update(kwargs)

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
    def categoryCount(cls):
        return len(cls.imageset_paths.keys())

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

class TestViews(BaseViewsTest):
    """
    Tests which don't require an imageset or a dataset
    """
    def test_page_dataset_new(self):
        rv = self.app.get('/datasets/images/classification/new')
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        assert 'New Image Classification Dataset' in rv.data, 'unexpected page format'

    def test_nonexistent_dataset(self):
        assert not self.dataset_exists('foo'), "dataset shouldn't exist"


class TestCreation(BaseViewsTestWithImageset):
    """
    Dataset creation tests
    """
    def test_nonexistent_folder(self):
        try:
            job_id = self.create_dataset(
                    folder_train = '/not-a-directory'
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

    def test_textfiles(self):
        for absolute_path in (True, False):
            for local_path in (True, False):
                yield self.check_textfiles, absolute_path, local_path

    def check_textfiles(self, absolute_path=True, local_path=True):
        """
        Create a dataset from textfiles

        Arguments:
        absolute_path -- if False, give relative paths and image folders
        """
        textfile_train_images = ''
        textfile_labels_file = ''
        label_id = 0
        for label, images in self.imageset_paths.iteritems():
            textfile_labels_file += '%s\n' % label
            for image in images:
                image_path = image
                if absolute_path:
                    image_path = os.path.join(self.imageset_folder, image_path)
                textfile_train_images += '%s %d\n' % (image_path, label_id)

            label_id += 1

        data = {
                'method': 'textfile',
                'textfile_use_val': 'y',
                }

        if local_path:
            train_file = os.path.join(self.imageset_folder, "local_train.txt")
            labels_file = os.path.join(self.imageset_folder, "local_labels.txt")
            # create files in local filesystem - these will be removed in tearDownClass() function
            with open(train_file, "w") as outfile:
                outfile.write(textfile_train_images)
            with open(labels_file, "w") as outfile:
                outfile.write(textfile_labels_file)
            data['textfile_use_local_files'] = 'True'
            data['textfile_local_train_images'] = train_file
            # Use the same file for training and validation.
            data['textfile_local_val_images'] = train_file
            data['textfile_local_labels_file'] = labels_file
        else:
            # StringIO wrapping is needed to simulate POST file upload.
            train_upload = (StringIO(textfile_train_images), "train.txt")
            # Use the same list for training and validation.
            val_upload = (StringIO(textfile_train_images), "val.txt")
            labels_upload = (StringIO(textfile_labels_file), "labels.txt")
            data['textfile_train_images'] = train_upload
            data['textfile_val_images'] = val_upload
            data['textfile_labels_file'] = labels_upload

        if not absolute_path:
            data['textfile_train_folder'] = self.imageset_folder
            data['textfile_val_folder'] = self.imageset_folder

        job_id = self.create_dataset(**data)
        assert self.dataset_wait_completion(job_id) == 'Done', 'create failed'

class TestImageCount(BaseViewsTestWithImageset):

    def test_image_count(self):
        for type in ['train','val','test']:
            yield self.check_image_count, type

    def check_image_count(self, type):
        data = {'folder_pct_val': 20,
                'folder_pct_test': 10}
        if type == 'val':
            data['has_val_folder'] = 'True'
            data['folder_val'] = self.imageset_folder
        elif type == 'test':
            data['has_test_folder'] = 'True'
            data['folder_test'] = self.imageset_folder

        job_id = self.create_dataset(**data)
        assert self.dataset_wait_completion(job_id) == 'Done', 'create failed'
        info = self.dataset_info(job_id)

        if type == 'train':
            assert len(info['ParseFolderTasks']) == 1, 'expected exactly one ParseFolderTasks'
            parse_info = info['ParseFolderTasks'][0]
            image_count = parse_info['train_count'] + parse_info['val_count'] + parse_info['test_count']
            assert parse_info['val_count'] == 0.2 * image_count
            assert parse_info['test_count'] == 0.1 * image_count
        else:
            assert len(info['ParseFolderTasks']) == 2, 'expected exactly one ParseFolderTasks'
            parse_info = info['ParseFolderTasks'][1]
            if type == 'val':
                assert parse_info['train_count'] == 0
                assert parse_info['test_count'] == 0
                image_count = parse_info['val_count']
            else:
                assert parse_info['train_count'] == 0
                assert parse_info['val_count'] == 0
                image_count = parse_info['test_count']
        assert self.categoryCount() == parse_info['label_count']
        assert image_count == DUMMY_IMAGE_COUNT * parse_info['label_count'], 'image count mismatch'
        assert self.delete_dataset(job_id) == 200, 'delete failed'
        assert not self.dataset_exists(job_id), 'dataset exists after delete'

class TestMaxPerClass(BaseViewsTestWithImageset):
    def test_max_per_class(self):
        for type in ['train','val','test']:
            yield self.check_max_per_class, type

    def check_max_per_class(self, type):
        # create dataset, asking for at most DUMMY_IMAGE_COUNT/2 images per class
        assert DUMMY_IMAGE_COUNT%2 == 0
        max_per_class = DUMMY_IMAGE_COUNT/2
        data = {'folder_pct_val': 0}
        if type == 'train':
            data['folder_train_max_per_class'] = max_per_class
        if type == 'val':
            data['has_val_folder'] = 'True'
            data['folder_val'] = self.imageset_folder
            data['folder_val_max_per_class'] = max_per_class
        elif type == 'test':
            data['has_test_folder'] = 'True'
            data['folder_test'] = self.imageset_folder
            data['folder_test_max_per_class'] = max_per_class

        job_id = self.create_dataset(**data)
        assert self.dataset_wait_completion(job_id) == 'Done', 'create failed'
        info = self.dataset_info(job_id)

        if type == 'train':
            assert len(info['ParseFolderTasks']) == 1, 'expected exactly one ParseFolderTasks'
            parse_info = info['ParseFolderTasks'][0]
        else:
            assert len(info['ParseFolderTasks']) == 2, 'expected exactly one ParseFolderTasks'
            parse_info = info['ParseFolderTasks'][1]

        image_count = parse_info['train_count'] + parse_info['val_count'] + parse_info['test_count']
        assert image_count == max_per_class * parse_info['label_count'], 'image count mismatch'
        assert self.delete_dataset(job_id) == 200, 'delete failed'
        assert not self.dataset_exists(job_id), 'dataset exists after delete'

class TestMinPerClass(BaseViewsTestWithImageset):

    UNBALANCED_CATEGORY = True

    def test_min_per_class(self):
        for type in ['train','val','test']:
            yield self.check_min_per_class, type

    def check_min_per_class(self, type):
        # create dataset, asking for one more image per class
        # than available in the "unbalanced" category
        min_per_class = DUMMY_IMAGE_COUNT/2+1
        data = {'folder_pct_val': 0}
        if type == 'train':
            data['folder_train_min_per_class'] = min_per_class
        if type == 'val':
            data['has_val_folder'] = 'True'
            data['folder_val'] = self.imageset_folder
            data['folder_val_min_per_class'] = min_per_class
        elif type == 'test':
            data['has_test_folder'] = 'True'
            data['folder_test'] = self.imageset_folder
            data['folder_test_min_per_class'] = min_per_class

        job_id = self.create_dataset(**data)
        assert self.dataset_wait_completion(job_id) == 'Done', 'create failed'
        info = self.dataset_info(job_id)

        if type == 'train':
            assert len(info['ParseFolderTasks']) == 1, 'expected exactly one ParseFolderTasks'
            parse_info = info['ParseFolderTasks'][0]
        else:
            assert len(info['ParseFolderTasks']) == 2, 'expected exactly two ParseFolderTasks'
            parse_info = info['ParseFolderTasks'][1]

        assert self.categoryCount() == parse_info['label_count']+1
        assert self.delete_dataset(job_id) == 200, 'delete failed'
        assert not self.dataset_exists(job_id), 'dataset exists after delete'


class TestCreated(BaseViewsTestWithDataset):
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

    def test_mean_dimensions(self):
        img_url = '/files/%s/mean.jpg' % self.dataset_id
        rv = self.app.get(img_url)
        assert rv.status_code == 200, 'GET on %s returned %s' % (img_url, rv.status_code)
        buff = StringIO(rv.data)
        buff.seek(0)
        pil_image = PIL.Image.open(buff)
        assert pil_image.size == (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), 'image size is %s' % (pil_image.size,)


class TestCreatedGrayscale(TestCreated):
    IMAGE_CHANNELS = 1

class TestCreatedWide(TestCreated):
    IMAGE_WIDTH = 20

class TestCreatedTall(TestCreated):
    IMAGE_HEIGHT = 20

