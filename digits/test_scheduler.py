# Copyright (c) 2014-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

from . import scheduler
from .config import config_value
from .job import Job
from .webapp import app
from digits import test_utils
from digits.utils import subclass, override


test_utils.skipIfNotFramework('none')


class TestScheduler():

    def get_scheduler(self):
        return scheduler.Scheduler(config_value('gpu_list'))

    def test_add_before_start(self):
        s = self.get_scheduler()
        assert not s.add_job(None), 'add_job should fail'

    def test_start_twice(self):
        s = self.get_scheduler()
        assert s.start(), 'failed to start'
        assert s.start(), 'failed to start the second time'
        assert s.stop(), 'failed to stop'

    def test_stop_before_start(self):
        s = self.get_scheduler()
        assert s.stop(), 'failed to stop'


@subclass
class JobForTesting(Job):

    @override
    def job_type(self):
        return 'Job For Testing'


class TestSchedulerFlow():

    @classmethod
    def setUpClass(cls):
        cls.s = scheduler.Scheduler(config_value('gpu_list'))
        assert cls.s.start(), 'failed to start'

    @classmethod
    def tearDownClass(cls):
        assert cls.s.stop(), 'failed to stop'

    def test_add_remove_job(self):
        with app.test_request_context():
            job = JobForTesting(name='testsuite-job', username='digits-testsuite')
            assert self.s.add_job(job), 'failed to add job'
            assert len(self.s.jobs) == 1, 'scheduler has %d jobs' % len(self.s.jobs)
            assert self.s.delete_job(job), 'failed to delete job'
            assert len(self.s.jobs) == 0, 'scheduler has %d jobs' % len(self.s.jobs)
