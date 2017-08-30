# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import json
import os
import shutil
import tempfile

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

from bs4 import BeautifulSoup
import PIL.Image

from .test_imageset_creator import create_classification_imageset
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
    # Inherited classes may want to override these default attributes
    IMAGE_COUNT = 10  # per class
    IMAGE_HEIGHT = 10
    IMAGE_WIDTH = 10
    IMAGE_CHANNELS = 3
    BACKEND = 'lmdb'
    ENCODING = 'png'
    COMPRESSION = 'none'

    UNBALANCED_CATEGORY = False

    @classmethod
    def setUpClass(cls):
        super(BaseViewsTestWithImageset, cls).setUpClass()
        cls.imageset_folder = tempfile.mkdtemp()
        # create imageset
        cls.imageset_paths = create_classification_imageset(
            cls.imageset_folder,
            image_count=cls.IMAGE_COUNT,
            add_unbalanced_category=cls.UNBALANCED_CATEGORY,
        )
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
            'group_name':       'test_group',
            'method':           'folder',
            'folder_train':     cls.imageset_folder,
            'resize_channels':  cls.IMAGE_CHANNELS,
            'resize_width':     cls.IMAGE_WIDTH,
            'resize_height':    cls.IMAGE_HEIGHT,
            'backend':          cls.BACKEND,
            'encoding':         cls.ENCODING,
            'compression':      cls.COMPRESSION,
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

    def test_clone(self):
        options_1 = {
            'encoding': 'png',
            'folder_pct_test': 0,
            'folder_pct_val': 25,
            'folder_test': '',
            'folder_test_max_per_class': None,
            'folder_test_min_per_class': 2,
            'folder_train_max_per_class': 3,
            'folder_train_min_per_class': 1,
            'folder_val_max_per_class': None,
            'folder_val_min_per_class': 2,
            'resize_mode': 'half_crop',
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

################################################################################
# Test classes
################################################################################


class TestViews(BaseViewsTest, test_utils.DatasetMixin):
    """
    Tests which don't require an imageset or a dataset
    """

    def test_page_dataset_new(self):
        rv = self.app.get('/datasets/images/classification/new')
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        assert 'New Image Classification Dataset' in rv.data, 'unexpected page format'

    def test_nonexistent_dataset(self):
        assert not self.dataset_exists('foo'), "dataset shouldn't exist"


class TestCreation(BaseViewsTestWithImageset, test_utils.DatasetMixin):
    """
    Dataset creation tests
    """

    def test_nonexistent_folder(self):
        try:
            self.create_dataset(
                folder_train='/not-a-directory'
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

    def test_abort_explore_fail(self):
        job_id = self.create_dataset()
        self.abort_dataset(job_id)
        rv = self.app.get('/datasets/images/classification/explore?job_id=%s&db=val' % job_id)
        assert rv.status_code == 500, 'page load should have failed'
        assert 'status should be' in rv.data, 'unexpected page format'


class TestImageCount(BaseViewsTestWithImageset, test_utils.DatasetMixin):

    def test_image_count(self):
        for type in ['train', 'val', 'test']:
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
        assert image_count == self.IMAGE_COUNT * parse_info['label_count'], 'image count mismatch'
        assert self.delete_dataset(job_id) == 200, 'delete failed'
        assert not self.dataset_exists(job_id), 'dataset exists after delete'


class TestMaxPerClass(BaseViewsTestWithImageset, test_utils.DatasetMixin):

    def test_max_per_class(self):
        for type in ['train', 'val', 'test']:
            yield self.check_max_per_class, type

    def check_max_per_class(self, type):
        # create dataset, asking for at most IMAGE_COUNT/2 images per class
        assert self.IMAGE_COUNT % 2 == 0
        max_per_class = self.IMAGE_COUNT / 2
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


class TestMinPerClass(BaseViewsTestWithImageset, test_utils.DatasetMixin):

    UNBALANCED_CATEGORY = True

    def test_min_per_class(self):
        for type in ['train', 'val', 'test']:
            yield self.check_min_per_class, type

    def check_min_per_class(self, type):
        # create dataset, asking for one more image per class
        # than available in the "unbalanced" category
        min_per_class = self.IMAGE_COUNT / 2 + 1
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

        assert self.categoryCount() == parse_info['label_count'] + 1
        assert self.delete_dataset(job_id) == 200, 'delete failed'
        assert not self.dataset_exists(job_id), 'dataset exists after delete'


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

    def test_mean_dimensions(self):
        img_url = '/files/%s/mean.jpg' % self.dataset_id
        rv = self.app.get(img_url)
        assert rv.status_code == 200, 'GET on %s returned %s' % (img_url, rv.status_code)
        buff = StringIO(rv.data)
        buff.seek(0)
        pil_image = PIL.Image.open(buff)
        assert pil_image.size == (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), 'image size is %s' % (pil_image.size,)

    def test_edit_name(self):
        status = self.edit_job(self.dataset_id,
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

    def test_backend_selection(self):
        rv = self.app.get('/datasets/%s.json' % self.dataset_id)
        content = json.loads(rv.data)
        for task in content['CreateDbTasks']:
            assert task['backend'] == self.BACKEND

    def test_explore_train(self):
        rv = self.app.get('/datasets/images/classification/explore?job_id=%s&db=train' % self.dataset_id)
        if self.BACKEND == 'hdf5':
            # Not supported yet
            assert rv.status_code == 500, 'page load should have failed'
            assert 'expected backend is lmdb' in rv.data, 'unexpected page format'
        else:
            assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
            assert 'Items per page' in rv.data, 'unexpected page format'

    def test_explore_val(self):
        rv = self.app.get('/datasets/images/classification/explore?job_id=%s&db=val' % self.dataset_id)
        if self.BACKEND == 'hdf5':
            # Not supported yet
            assert rv.status_code == 500, 'page load should have failed'
            assert 'expected backend is lmdb' in rv.data, 'unexpected page format'
        else:
            assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
            assert 'Items per page' in rv.data, 'unexpected page format'


class TestCreatedGrayscale(TestCreated, test_utils.DatasetMixin):
    IMAGE_CHANNELS = 1


class TestCreatedWide(TestCreated, test_utils.DatasetMixin):
    IMAGE_WIDTH = 20


class TestCreatedTall(TestCreated, test_utils.DatasetMixin):
    IMAGE_HEIGHT = 20


class TestCreatedJPEG(TestCreated, test_utils.DatasetMixin):
    ENCODING = 'jpg'


class TestCreatedRaw(TestCreated, test_utils.DatasetMixin):
    ENCODING = 'none'


class TestCreatedRawGrayscale(TestCreated, test_utils.DatasetMixin):
    ENCODING = 'none'
    IMAGE_CHANNELS = 1


class TestCreatedHdf5(TestCreated, test_utils.DatasetMixin):
    BACKEND = 'hdf5'

    def test_compression_method(self):
        rv = self.app.get('/datasets/%s.json' % self.dataset_id)
        content = json.loads(rv.data)
        for task in content['CreateDbTasks']:
            assert task['compression'] == self.COMPRESSION


class TestCreatedHdf5Gzip(TestCreatedHdf5, test_utils.DatasetMixin):
    COMPRESSION = 'gzip'
