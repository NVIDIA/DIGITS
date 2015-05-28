# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

from nose.tools import assert_raises
import mock

from . import scheduler as _
from config import config_value
from job import Job

class TestScheduler():

    def get_scheduler(self):
        return _.Scheduler(config_value('gpu_list'))

    def test_add_before_start(self):
        """add_job before scheduler start"""
        s = self.get_scheduler()
        assert not s.add_job(None), 'add_job should fail'

    def test_start_twice(self):
        """start scheduler twice"""
        s = self.get_scheduler()
        assert s.start(), 'failed to start'
        assert s.start(), 'failed to start the second time'
        assert s.stop(), 'failed to stop'

    def test_stop_before_start(self):
        """stop scheduler before start"""
        s = self.get_scheduler()
        assert s.stop(), 'failed to stop'


class TestSchedulerFlow():

    @classmethod
    def setUpClass(cls):
        cls.s = _.Scheduler(config_value('gpu_list'))
        assert cls.s.start(), 'failed to start'

    @classmethod
    def tearDownClass(cls):
        assert cls.s.stop(), 'failed to stop'

    def test_add_remove_job(self):
        """add and remove a job"""
        job = Job('tmp')
        assert self.s.add_job(job), 'failed to add job'
        assert len(self.s.jobs) == 1, 'scheduler has %d jobs' % len(self.s.jobs)
        assert self.s.delete_job(job), 'failed to delete job'
        assert len(self.s.jobs) == 0, 'scheduler has %d jobs' % len(self.s.jobs)

