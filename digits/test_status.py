# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

from gevent import monkey; monkey.patch_all()
from nose.tools import assert_raises
import mock
import pickle
import tempfile

from status import Status
from config import config_value
from job import Job

class TestScheduler():

    def test_run_too_soon(self):
        job = Job('test')
        job.status = Status.WAIT
        job.status = Status.RUN
        # Status.WAIT should be removed so the len should be 2 rather
        # than 3.
        assert len(job.status_history) == 2, 'history length should be 2'

    def test_empty_history(self):
        job = Job('test')

        job.status = Status.WAIT
        job.status = Status.RUN
        job.status_history = []
        # An empty history should not happen, but if it did, the value
        # should be Status.INIT.
        assert job.status == Status.INIT, 'status should be Status.INIT'

    def test_set_dict(self):
        job = Job('test')

        # testing some untested cases in set_dict()
        job.status = Status.ERROR
        assert job.status.css == 'danger', 'status.css should be "danger".'

        job.status = '404'
        assert job.status.css == 'default', 'status.css should be "default".'

    def test_equality(self):
        s = Status(Status.INIT)

        # testing __eq__
        assert (s == Status.INIT), 'should be true.'
        assert (s == 'I'), 'should be true.'
        assert not (s == 7), 'should be false.'

        assert not (s != Status.INIT), 'should be false.'
        assert not (s != 'I'), 'should be false.'
        assert (s != 7), 'should be true.'


    def test_pickle(self):
        # Testng __setstate__ and __getstate__

        s = Status(Status.INIT)
        s = Status.WAIT

        loaded_status = None

        tmpfile_path = 'tmp.p'
        with open(tmpfile_path, 'wb') as tmpfile:
            pickle.dump(s, tmpfile)
            tmpfile.close()

        with open(tmpfile_path, 'rb') as tmpfile:
            loaded_status = pickle.load(tmpfile)
            tmpfile.close()

        print loaded_status

        assert loaded_status == Status.WAIT, 'status should be WAIT'

    def test_str(self):
        # Testing __str__
        s = Status(Status.INIT)
        s = Status.WAIT

        assert str(s) == 'W', 'should be W'

