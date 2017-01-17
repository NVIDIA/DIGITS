# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import json
import time
import urllib

from urlparse import urlparse

from . import test_utils
from . import webapp

################################################################################
# Base classes (they don't start with "Test" so nose won't run them)
################################################################################


class BaseViewsTest(object):
    """
    Abstract class with a Flask context and a Scheduler
    Provides some other useful functions to children who inherit this
    """
    @classmethod
    def setUpClass(cls):
        # Call super.setUpClass() unless we're the last in the class hierarchy
        supercls = super(BaseViewsTest, cls)
        if hasattr(supercls, 'setUpClass'):
            supercls.setUpClass()

        # Start up the server
        assert webapp.scheduler.start(), "scheduler wouldn't start"
        webapp.app.config['WTF_CSRF_ENABLED'] = False
        webapp.app.config['TESTING'] = True
        cls.app = webapp.app.test_client()
        cls.created_datasets = []
        cls.created_models = []

        rv = cls.app.post('/login', data={
            'username': 'digits-testsuite'})
        assert rv.status_code == 302, 'Login failed with %s' % rv.status_code

    @classmethod
    def tearDownClass(cls):
        # Remove all created jobs
        for job_id in cls.created_models:
            cls.delete_model(job_id)
        for job_id in cls.created_datasets:
            cls.delete_dataset(job_id)

        rv = cls.app.post('/logout')
        assert rv.status_code == 302, 'Logout failed with %s' % rv.status_code

    @classmethod
    def job_id_from_response(cls, rv):
        """
        Extract the job_id from an HTTP response
        """
        job_url = rv.headers['Location']
        parsed_url = urlparse(job_url)
        return parsed_url.path.split('/')[-1]

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
    def job_info(cls, job_id, job_type='jobs'):
        """
        Get job information (full JSON response)
        """
        url = '/%s/%s.json' % (job_type, job_id)
        rv = cls.app.get(url)
        assert rv.status_code == 200, 'Cannot get info from job %s. "%s" returned %s' % (job_id, url, rv.status_code)
        info = json.loads(rv.data)
        return info

    @classmethod
    def job_info_html(cls, job_id, job_type='jobs'):
        """
        Get job information (full HTML response)
        """
        url = '/%s/%s' % (job_type, job_id)
        rv = cls.app.get(url)
        assert rv.status_code == 200, 'Cannot get info from job %s. "%s" returned %s' % (job_id, url, rv.status_code)
        return rv.data

    @classmethod
    def abort_job(cls, job_id, job_type='jobs'):
        """
        Abort a job
        Returns the HTTP status code
        """
        rv = cls.app.post('/%s/%s/abort' % (job_type, job_id))
        return rv.status_code

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
                # make sure job appears in completed jobs
                url = '/completed_jobs.json'
                rv = cls.app.get(url)
                assert rv.status_code == 200, 'Cannot get info from job %s. "%s" returned %s' % (
                    job_id, url, rv.status_code)
                info = json.loads(rv.data)
                dataset_ids = [job['id'] for job in info['datasets']]
                model_ids = [job['id'] for job in info['models']]
                assert job_id in dataset_ids or job_id in model_ids, "job %s not found in completed jobs" % job_id
                # make sure job can be shown without error
                url = '/jobs/%s' % job_id
                rv = cls.app.get(url, follow_redirects=True)
                assert rv.status_code == 200, 'Cannot get info from job %s. "%s" returned %s' % (
                    job_id, url, rv.status_code)
                assert job_id in rv.data
                return status
            assert (time.time() - start) < timeout, 'Job took more than %s seconds' % timeout
            time.sleep(polling_period)

    @classmethod
    def edit_job(cls, job_id, name=None, notes=None):
        """
        Edit the name of a job
        """
        data = {}
        if name:
            data['job_name'] = name
        if notes:
            data['job_notes'] = notes
        rv = cls.app.put('/jobs/%s' % job_id, data=data)
        return rv.status_code

    @classmethod
    def delete_job(cls, job_id, job_type='jobs'):
        """
        Delete a job
        Returns the HTTP status code
        """
        rv = cls.app.delete('/%s/%s' % (job_type, job_id))
        return rv.status_code

################################################################################
# Test classes
################################################################################


class TestViews(BaseViewsTest):

    @classmethod
    def setUpClass(cls):
        test_utils.skipIfNotFramework('none')
        super(TestViews, cls).setUpClass()

    def test_homepage(self):
        rv = self.app.get('/')
        assert rv.status_code == 200, 'page load failed with %s' % rv.status_code
        for text in ['Home', 'Datasets', 'Models']:
            assert text in rv.data, 'unexpected page format'

    def test_invalid_page(self):
        rv = self.app.get('/foo')
        assert rv.status_code == 404, 'should return 404'

    def test_autocomplete(self):
        for absolute_path in (True, False):
            yield self.check_autocomplete, absolute_path

    def check_autocomplete(self, absolute_path):
        path = '/' if absolute_path else './'
        url = '/autocomplete/path?query=%s' % (urllib.quote(path, safe=''))
        rv = self.app.get(url)
        assert rv.status_code == 200
        status = json.loads(rv.data)
        assert 'suggestions' in status
