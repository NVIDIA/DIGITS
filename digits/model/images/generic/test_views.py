# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import itertools
import json
import numpy as np
import os
import PIL.Image
import tempfile
import time
import unittest

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

from bs4 import BeautifulSoup

from digits import extensions
from digits.config import config_value
import digits.dataset.images.generic.test_views
import digits.dataset.generic.test_views
import digits.test_views
from digits import test_utils
import digits.webapp


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
  name: "loss"
  type: "EuclideanLoss"
  bottom: "output"
  bottom: "label"
  top: "loss"
  exclude { stage: "deploy" }
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
    -- set all weights and biases to zero as this speeds learning up
    -- for the type of problem we're trying to solve in this test
    local linearLayer = nn.Linear(nDim, 2)
    linearLayer.weight:fill(0)
    linearLayer.bias:fill(0)
    net:add(linearLayer) -- c*h*w -> 2
    return {
        model = net,
        loss = nn.MSECriterion(),
    }
end
"""

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
        return cls.TORCH_NETWORK if cls.FRAMEWORK == 'torch' else cls.CAFFE_NETWORK


class BaseViewsTestWithAnyDataset(BaseViewsTest):
    """
    Provides a dataset
    This is a common interface to work with either "images/generic"
    datasets or "generic" datasets. The dataset type to use is chosen
    further down in the class hierarchy, see e.g. BaseViewsTestWithDataset
    """

    # Inherited classes may want to override these attributes
    CROP_SIZE = None
    TRAIN_EPOCHS = 3
    LR_POLICY = None
    LEARNING_RATE = None
    BATCH_SIZE = 10

    @classmethod
    def setUpClass(cls, **kwargs):
        super(BaseViewsTestWithAnyDataset, cls).setUpClass(**kwargs)
        cls.created_models = []

    @classmethod
    def tearDownClass(cls):
        # delete any created datasets
        for job_id in cls.created_models:
            cls.delete_model(job_id)
        super(BaseViewsTestWithAnyDataset, cls).tearDownClass()

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
            'group_name':       'test_group',
            'dataset':          cls.dataset_id,
            'method':           'custom',
            'custom_network':   cls.network(),
            'batch_size':       cls.BATCH_SIZE,
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
            data = json.loads(rv.data)
            if 'jobs' in data.keys():
                return [j['id'] for j in data['jobs']]
            else:
                return data['id']

        # expect a redirect
        if not 300 <= rv.status_code <= 310:
            print 'Status code:', rv.status_code
            s = BeautifulSoup(rv.data, 'html.parser')
            div = s.select('div.alert-danger')
            if div:
                print div[0]
            else:
                print rv.data
            raise RuntimeError('Failed to create dataset - status %s' % rv.status_code)

        job_id = cls.job_id_from_response(rv)
        assert cls.model_exists(job_id), 'model not found after successful creation'

        cls.created_models.append(job_id)
        return job_id


class BaseViewsTestWithDataset(BaseViewsTestWithAnyDataset,
                               digits.dataset.images.generic.test_views.BaseViewsTestWithDataset):
    """
    This inherits from BaseViewsTestWithAnyDataset and
    digits.dataset.images.generic.test_views.BaseViewsTestWithDataset
    in order to provide an interface to test models on "images/generic" datasets
    """
    pass


class BaseViewsTestWithModelWithAnyDataset(BaseViewsTestWithAnyDataset):
    """
    Provides a model
    """
    @classmethod
    def setUpClass(cls, **kwargs):
        use_mean = kwargs.pop('use_mean', None)
        super(BaseViewsTestWithModelWithAnyDataset, cls).setUpClass(**kwargs)
        cls.model_id = cls.create_model(json=True, use_mean=use_mean)
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

    def test_view_config(self):
        extension = extensions.view.get_default_extension()
        rv = self.app.get('/models/view-config/%s' % extension.get_id())
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code

    def test_visualize_network(self):
        rv = self.app.post('/models/visualize-network?framework=' + self.FRAMEWORK,
                           data={'custom_network': self.network()}
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
        not config_value('caffe')['cuda_enabled'],
        'CUDA disabled')
    @unittest.skipIf(
        config_value('caffe')['multi_gpu'],
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
        not config_value('caffe')['cuda_enabled'],
        'CUDA disabled')
    @unittest.skipIf(
        not config_value('caffe')['multi_gpu'],
        'multi-GPU disabled')
    def test_select_gpus(self):
        # test all possible combinations
        gpu_list = config_value('gpu_list').split(',')
        for i in xrange(len(gpu_list)):
            for combination in itertools.combinations(gpu_list, i + 1):
                yield self.check_select_gpus, combination

    def check_select_gpus(self, gpu_list):
        job_id = self.create_model(select_gpus_list=','.join(gpu_list), batch_size=len(gpu_list))
        assert self.model_wait_completion(job_id) == 'Done', 'create failed'

    def infer_one_for_job(self, job_id):
        # carry out one inference test per category in dataset
        image_path = os.path.join(self.imageset_folder, self.test_image)
        with open(image_path, 'rb') as infile:
            # StringIO wrapping is needed to simulate POST file upload.
            image_upload = (StringIO(infile.read()), 'image.png')

        rv = self.app.post(
            '/models/images/generic/infer_one?job_id=%s' % job_id,
            data={
                'image_file': image_upload,
                'show_visualizations': 'y',
            }
        )
        s = BeautifulSoup(rv.data, 'html.parser')
        body = s.select('body')
        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)

    def test_infer_one_mean_image(self):
        # test the creation
        job_id = self.create_model(use_mean='image')
        assert self.model_wait_completion(job_id) == 'Done', 'job failed'
        self.infer_one_for_job(job_id)

    def test_infer_one_mean_pixel(self):
        # test the creation
        job_id = self.create_model(use_mean='pixel')
        assert self.model_wait_completion(job_id) == 'Done', 'job failed'
        self.infer_one_for_job(job_id)

    def test_infer_one_mean_none(self):
        # test the creation
        job_id = self.create_model(use_mean='none')
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

        # Clone job1 as job2
        options_2 = {
            'clone': job1_id,
        }

        job2_id = self.create_model(**options_2)
        assert self.model_wait_completion(job2_id) == 'Done', 'second job failed'
        rv = self.app.get('/models/%s.json' % job2_id)
        assert rv.status_code == 200, 'json load failed with %s' % rv.status_code
        content2 = json.loads(rv.data)

        # These will be different
        content1.pop('id')
        content2.pop('id')
        content1.pop('directory')
        content2.pop('directory')
        content1.pop('creation time')
        content2.pop('creation time')
        content1.pop('job id')
        content2.pop('job id')

        assert (content1 == content2), 'job content does not match'

        job1 = digits.webapp.scheduler.get_job(job1_id)
        job2 = digits.webapp.scheduler.get_job(job2_id)

        assert (job1.form_data == job2.form_data), 'form content does not match'


class BaseTestCreatedWithAnyDataset(BaseViewsTestWithModelWithAnyDataset):
    """
    Tests on a model that has already been created
    """

    def test_save(self):
        job = digits.webapp.scheduler.get_job(self.model_id)
        assert job.save(), 'Job failed to save'

    def test_get_snapshot(self):
        job = digits.webapp.scheduler.get_job(self.model_id)
        task = job.train_task()
        f = task.get_snapshot(-1)

        assert f, "Failed to load snapshot"
        filename = task.get_snapshot_filename(-1)
        assert filename, "Failed to get filename"

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
        with open(image_path, 'rb') as infile:
            # StringIO wrapping is needed to simulate POST file upload.
            image_upload = (StringIO(infile.read()), 'image.png')

        rv = self.app.post(
            '/models/images/generic/infer_one?job_id=%s' % self.model_id,
            data={
                'image_file': image_upload,
                'show_visualizations': 'y',
            }
        )
        s = BeautifulSoup(rv.data, 'html.parser')
        body = s.select('body')
        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)

    def test_infer_one_json(self):
        image_path = os.path.join(self.imageset_folder, self.test_image)
        with open(image_path, 'rb') as infile:
            # StringIO wrapping is needed to simulate POST file upload.
            image_upload = (StringIO(infile.read()), 'image.png')

        rv = self.app.post(
            '/models/images/generic/infer_one.json?job_id=%s' % self.model_id,
            data={
                'image_file': image_upload,
            }
        )
        assert rv.status_code == 200, 'POST failed with %s' % rv.status_code
        data = json.loads(rv.data)
        assert data['outputs']['output'][0][0] > 0 and \
            data['outputs']['output'][0][1] > 0, \
            'image regression result is wrong: %s' % data['outputs']['output']

    def test_infer_many(self):
        # use the same image twice to make a list of two images
        textfile_images = '%s\n%s\n' % (self.test_image, self.test_image)

        # StringIO wrapping is needed to simulate POST file upload.
        file_upload = (StringIO(textfile_images), 'images.txt')

        rv = self.app.post(
            '/models/images/generic/infer_many?job_id=%s' % self.model_id,
            data={'image_list': file_upload}
        )
        s = BeautifulSoup(rv.data, 'html.parser')
        body = s.select('body')
        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)
        headers = s.select('table.table th')
        assert headers is not None, 'unrecognized page format'

    def test_infer_db(self):
        if self.val_db_path is None:
            raise unittest.SkipTest('Class has no validation db')
        rv = self.app.post(
            '/models/images/generic/infer_db?job_id=%s' % self.model_id,
            data={'db_path': self.val_db_path}
        )
        s = BeautifulSoup(rv.data, 'html.parser')
        body = s.select('body')
        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, body)
        headers = s.select('table.table th')
        assert headers is not None, 'unrecognized page format'

    def test_infer_many_from_folder(self):
        textfile_images = '%s\n' % os.path.basename(self.test_image)

        # StringIO wrapping is needed to simulate POST file upload.
        file_upload = (StringIO(textfile_images), 'images.txt')

        # try selecting the extension explicitly
        extension = extensions.view.get_default_extension()
        extension_id = extension.get_id()

        rv = self.app.post(
            '/models/images/generic/infer_many?job_id=%s' % self.model_id,
            data={'image_list': file_upload,
                  'image_folder': os.path.dirname(self.test_image),
                  'view_extension_id': extension_id}
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
            data={'image_list': file_upload}
        )
        assert rv.status_code == 200, 'POST failed with %s' % rv.status_code
        data = json.loads(rv.data)
        assert 'outputs' in data, 'invalid response'

    def test_infer_db_json(self):
        if self.val_db_path is None:
            raise unittest.SkipTest('Class has no validation db')
        rv = self.app.post(
            '/models/images/generic/infer_db.json?job_id=%s' % self.model_id,
            data={'db_path': self.val_db_path}
        )
        assert rv.status_code == 200, 'POST failed with %s\n\n%s' % (rv.status_code, rv.data)
        data = json.loads(rv.data)
        assert 'outputs' in data, 'invalid response'


class BaseTestCreated(BaseTestCreatedWithAnyDataset,
                      digits.dataset.images.generic.test_views.BaseViewsTestWithDataset):
    """
    Tests on a model that has already been created with an "images/generic" dataset
    """
    pass


class BaseTestCreatedWithGradientDataExtension(BaseTestCreatedWithAnyDataset,
                                               digits.dataset.generic.test_views.BaseViewsTestWithDataset):
    """
    Tests on a model that has already been created with a "generic" dataset,
    using the gradients extension in that instance
    """
    EXTENSION_ID = "image-gradients"

    @classmethod
    def setUpClass(cls, **kwargs):
        if not hasattr(cls, 'imageset_folder'):
            # Create test image
            cls.imageset_folder = tempfile.mkdtemp()
            image_width = cls.IMAGE_WIDTH
            image_height = cls.IMAGE_HEIGHT
            yy, xx = np.mgrid[:image_height,
                              :image_width].astype('float')
            xslope, yslope = 0.5, 0.5
            a = xslope * 255 / image_width
            b = yslope * 255 / image_height
            test_image = a * (xx - image_width / 2) + b * (yy - image_height / 2) + 127.5
            test_image = test_image.astype('uint8')
            pil_img = PIL.Image.fromarray(test_image)
            cls.test_image = os.path.join(cls.imageset_folder, 'test.png')
            pil_img.save(cls.test_image)
        # note: model created in BaseTestCreatedWithAnyDataset.setUpClass method
        super(BaseTestCreatedWithGradientDataExtension, cls).setUpClass()

    def test_infer_extension_json(self):
        rv = self.app.post(
            '/models/images/generic/infer_extension.json?job_id=%s' % self.model_id,
            data={
                'gradient_x': 0.5,
                'gradient_y': -0.5,
            }
        )
        assert rv.status_code == 200, 'POST failed with %s' % rv.status_code
        data = json.loads(rv.data)
        output = data['outputs'][data['outputs'].keys()[0]]['output']
        assert output[0] > 0 and \
            output[1] < 0, \
            'image regression result is wrong: %s' % data['outputs']['output']


class BaseTestCreatedWithImageProcessingExtension(
        BaseTestCreatedWithAnyDataset,
        digits.dataset.generic.test_views.BaseViewsTestWithDataset):
    """
    Test Image processing extension with a dummy identity network
    """

    CAFFE_NETWORK = \
        """
layer {
  name: "identity"
  type: "Power"
  bottom: "data"
  top: "output"
}
layer {
  name: "loss"
  type: "EuclideanLoss"
  bottom: "output"
  bottom: "label"
  top: "loss"
  exclude { stage: "deploy" }
}
"""

    TORCH_NETWORK = \
        """
return function(p)
    return {
        -- simple identity network
        model = nn.Sequential():add(nn.Identity()),
        loss = nn.MSECriterion(),
    }
end
"""

    EXTENSION_ID = "image-processing"
    VARIABLE_SIZE_DATASET = False
    NUM_IMAGES = 100
    MEAN = 'none'

    @classmethod
    def setUpClass(cls, **kwargs):
        if cls.VARIABLE_SIZE_DATASET:
            cls.BATCH_SIZE = 1
            cls.create_variable_size_random_imageset(
                num_images=cls.NUM_IMAGES)
        else:
            cls.create_random_imageset(
                num_images=cls.NUM_IMAGES,
                image_width=cls.IMAGE_WIDTH,
                image_height=cls.IMAGE_HEIGHT)
        super(BaseTestCreatedWithImageProcessingExtension, cls).setUpClass(
            feature_folder=cls.imageset_folder,
            label_folder=cls.imageset_folder,
            channel_conversion='L',
            dsopts_force_same_shape='0' if cls.VARIABLE_SIZE_DATASET else '1',
            use_mean=cls.MEAN)

    def test_infer_one_json(self):
        image_path = os.path.join(self.imageset_folder, self.test_image)
        with open(image_path, 'rb') as infile:
            # StringIO wrapping is needed to simulate POST file upload.
            image_upload = (StringIO(infile.read()), 'image.png')

        rv = self.app.post(
            '/models/images/generic/infer_one.json?job_id=%s' % self.model_id,
            data={'image_file': image_upload}
        )
        assert rv.status_code == 200, 'POST failed with %s' % rv.status_code
        data = json.loads(rv.data)
        data_shape = np.array(data['outputs']['output']).shape
        if not self.VARIABLE_SIZE_DATASET:
            assert data_shape == (1, self.CHANNELS, self.IMAGE_WIDTH, self.IMAGE_HEIGHT)

    def test_infer_one_noresize_json(self):
        # create large random image
        shape = (self.CHANNELS, 10 * self.IMAGE_HEIGHT, 5 * self.IMAGE_WIDTH)
        x = np.random.randint(
            low=0,
            high=256,
            size=shape)
        if self.CHANNELS == 1:
            # drop channel dimension
            x = x[0]
        x = x.astype('uint8')
        pil_img = PIL.Image.fromarray(x)
        # create output stream
        s = StringIO()
        pil_img.save(s, format="png")
        # create input stream
        s = StringIO(s.getvalue())
        image_upload = (s, 'image.png')
        # post request
        rv = self.app.post(
            '/models/images/generic/infer_one.json?job_id=%s' % self.model_id,
            data={'image_file': image_upload, 'dont_resize': 'y'}
        )
        assert rv.status_code == 200, 'POST failed with %s' % rv.status_code
        data = json.loads(rv.data)
        data_shape = np.array(data['outputs']['output']).shape
        assert data_shape == (1,) + shape

    def test_infer_db(self):
        if self.VARIABLE_SIZE_DATASET:
            raise unittest.SkipTest('Skip variable-size inference test')
        super(BaseTestCreatedWithImageProcessingExtension, self).test_infer_db()

    def test_infer_db_json(self):
        if self.VARIABLE_SIZE_DATASET:
            raise unittest.SkipTest('Skip variable-size inference test')
        super(BaseTestCreatedWithImageProcessingExtension, self).test_infer_db_json()


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
            self.create_model(dataset=dataset_id)
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
  name: "loss"
  type: "EuclideanLoss"
  bottom: "output"
  bottom: "label"
  top: "loss"
  exclude { stage: "deploy" }
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
    -- set all weights and biases to zero as this speeds learning up
    -- for the type of problem we're trying to solve in this test
    local linearLayer = nn.Linear(channels*croplen*croplen, 2)
    linearLayer.weight:fill(0)
    linearLayer.bias:fill(0)
    net:add(linearLayer) -- c*croplen*croplen -> 2
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


class TestCaffeViews(BaseTestViews, test_utils.CaffeMixin):
    pass


class TestCaffeCreation(BaseTestCreation, test_utils.CaffeMixin):
    pass


class TestCaffeCreated(BaseTestCreated, test_utils.CaffeMixin):
    pass


class TestCaffeCreatedWithGradientDataExtension(
        BaseTestCreatedWithGradientDataExtension, test_utils.CaffeMixin):
    pass


class TestCaffeCreatedWithGradientDataExtensionNoValSet(
        BaseTestCreatedWithGradientDataExtension, test_utils.CaffeMixin):

    @classmethod
    def setUpClass(cls):
        super(TestCaffeCreatedWithGradientDataExtensionNoValSet, cls).setUpClass(val_image_count=0)


class TestCaffeCreatedWithImageProcessingExtensionMeanImage(
        BaseTestCreatedWithImageProcessingExtension, test_utils.CaffeMixin):
    MEAN = 'image'


class TestCaffeCreatedWithImageProcessingExtensionMeanPixel(
        BaseTestCreatedWithImageProcessingExtension, test_utils.CaffeMixin):
    MEAN = 'pixel'


class TestCaffeCreatedWithImageProcessingExtensionMeanNone(
        BaseTestCreatedWithImageProcessingExtension, test_utils.CaffeMixin):
    MEAN = 'none'


class TestCaffeCreatedVariableSizeDataset(
        BaseTestCreatedWithImageProcessingExtension, test_utils.CaffeMixin):
    MEAN = 'none'
    VARIABLE_SIZE_DATASET = True


class TestCaffeDatasetModelInteractions(BaseTestDatasetModelInteractions, test_utils.CaffeMixin):
    pass


class TestCaffeCreatedCropInNetwork(BaseTestCreatedCropInNetwork, test_utils.CaffeMixin):
    pass


class TestCaffeCreatedCropInForm(BaseTestCreatedCropInForm, test_utils.CaffeMixin):
    pass


class TestTorchViews(BaseTestViews, test_utils.TorchMixin):
    pass


class TestTorchCreation(BaseTestCreation, test_utils.TorchMixin):
    pass


class TestTorchCreated(BaseTestCreated, test_utils.TorchMixin):
    pass


class TestTorchCreatedWithGradientDataExtension(
        BaseTestCreatedWithGradientDataExtension, test_utils.TorchMixin):
    pass


class TestTorchCreatedWithGradientDataExtensionNoValSet(
        BaseTestCreatedWithGradientDataExtension, test_utils.TorchMixin):

    @classmethod
    def setUpClass(cls):
        super(TestTorchCreatedWithGradientDataExtensionNoValSet, cls).setUpClass(val_image_count=0)


class TestTorchCreatedWithImageProcessingExtensionMeanImage(
        BaseTestCreatedWithImageProcessingExtension, test_utils.TorchMixin):
    MEAN = 'image'


class TestTorchCreatedWithImageProcessingExtensionMeanPixel(
        BaseTestCreatedWithImageProcessingExtension, test_utils.TorchMixin):
    MEAN = 'pixel'


class TestTorchCreatedWithImageProcessingExtensionMeanNone(
        BaseTestCreatedWithImageProcessingExtension, test_utils.TorchMixin):
    MEAN = 'none'


class TestTorchCreatedVariableSizeDataset(
        BaseTestCreatedWithImageProcessingExtension, test_utils.TorchMixin):
    MEAN = 'none'
    VARIABLE_SIZE_DATASET = True


class TestTorchCreatedCropInNetwork(BaseTestCreatedCropInNetwork, test_utils.TorchMixin):
    pass


class TestTorchCreatedCropInForm(BaseTestCreatedCropInForm, test_utils.TorchMixin):
    pass


class TestTorchDatasetModelInteractions(BaseTestDatasetModelInteractions, test_utils.TorchMixin):
    pass


class TestTorchTableOutput(BaseTestCreated, test_utils.TorchMixin):
    TORCH_NETWORK = \
        """
return function(p)
    -- same network as in class BaseTestCreated except that each gradient
    -- is learnt separately: the input is fed into nn.ConcatTable and
    -- each branch outputs one of the gradients
    local nDim = 1
    if p.inputShape then p.inputShape:apply(function(x) nDim=nDim*x end) end
    local net = nn.Sequential()
    net:add(nn.MulConstant(0.004))
    net:add(nn.View(-1):setNumInputDims(3))  -- flatten
    -- set all weights and biases to zero as this speeds learning up
    -- for the type of problem we're trying to solve in this test
    local linearLayer1 = nn.Linear(nDim, 1)
    linearLayer1.weight:fill(0)
    linearLayer1.bias:fill(0)
    local linearLayer2 = nn.Linear(nDim, 1)
    linearLayer2.weight:fill(0)
    linearLayer2.bias:fill(0)
    -- create concat table
    local parallel = nn.ConcatTable()
    parallel:add(linearLayer1):add(linearLayer2)
    net:add(parallel)
    -- create two MSE criteria to optimize each gradient separately
    local mse1 = nn.MSECriterion()
    local mse2 = nn.MSECriterion()
    -- now create a criterion that takes as input each of the two criteria
    local finalCriterion = nn.ParallelCriterion(false):add(mse1):add(mse2)
    -- create label hook
    function labelHook(input, dblabel)
        -- split label alongside 2nd dimension
        local labelTable = dblabel:split(1,2)
        return labelTable
    end
    return {
        model = net,
        loss = finalCriterion,
        labelHook = labelHook,
    }
end
"""


class TestTorchNDOutput(BaseTestCreated, test_utils.TorchMixin):
    CROP_SIZE = 8
    TORCH_NETWORK = \
        """
return function(p)
    -- this model just forwards the input as is
    local net = nn.Sequential():add(nn.Identity())
    -- create label hook
    function labelHook(input, dblabel)
        return input
    end
    return {
        model = net,
        loss = nn.AbsCriterion(),
        labelHook = labelHook,
    }
end
"""

    def test_infer_one_json(self):

        image_path = os.path.join(self.imageset_folder, self.test_image)
        with open(image_path, 'rb') as infile:
            # StringIO wrapping is needed to simulate POST file upload.
            image_upload = (StringIO(infile.read()), 'image.png')

        rv = self.app.post(
            '/models/images/generic/infer_one.json?job_id=%s' % self.model_id,
            data={
                'image_file': image_upload,
            }
        )
        assert rv.status_code == 200, 'POST failed with %s' % rv.status_code
        # make sure the shape of the output matches the shape of the input
        data = json.loads(rv.data)
        output = np.array(data['outputs']['output'][0])
        assert output.shape == (1, self.CROP_SIZE, self.CROP_SIZE), \
            'shape mismatch: %s' % str(output.shape)


class TestSweepCreation(BaseViewsTestWithDataset, test_utils.CaffeMixin):
    """
    Model creation tests
    """

    def test_sweep(self):
        job_ids = self.create_model(json=True, learning_rate='[0.01, 0.02]', batch_size='[8, 10]')
        for job_id in job_ids:
            assert self.model_wait_completion(job_id) == 'Done', 'create failed'
            assert self.delete_model(job_id) == 200, 'delete failed'
            assert not self.model_exists(job_id), 'model exists after delete'


class TestAllInOneNetwork(BaseTestCreation, BaseTestCreated, test_utils.CaffeMixin):
    """
    Test an all-in-one network
    """
    CAFFE_NETWORK = \
        """
layer {
  name: "train_data"
  type: "Data"
  top: "scaled_data"
  transform_param {
    scale: 0.004
  }
  include { phase: TRAIN }
}
layer {
  name: "train_label"
  type: "Data"
  top: "label"
  include { phase: TRAIN }
}
layer {
  name: "val_data"
  type: "Data"
  top: "scaled_data"
  transform_param {
    scale: 0.004
  }
  include { phase: TEST }
}
layer {
  name: "val_label"
  type: "Data"
  top: "label"
  include { phase: TEST }
}
layer {
  name: "scale"
  type: "Power"
  bottom: "data"
  top: "scaled_data"
  power_param {
    scale: 0.004
  }
  include { stage: "deploy" }
}
layer {
  name: "hidden"
  type: "InnerProduct"
  bottom: "scaled_data"
  top: "output"
  inner_product_param {
    num_output: 2
  }
}
layer {
  name: "loss"
  type: "EuclideanLoss"
  bottom: "output"
  bottom: "label"
  top: "loss"
  exclude { stage: "deploy" }
}
"""
