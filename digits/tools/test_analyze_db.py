# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.

import os.path
import shutil
import tempfile

import lmdb
import numpy as np

from . import analyze_db
from digits import test_utils

# Must import after importing digits.config
import caffe.io


test_utils.skipIfNotFramework('none')


class BaseTestWithDB(object):
    SAME_SHAPE = True
    PASS_DEFAULTS = True
    PASS_FORCE = True
    PASS_COUNT = True

    @classmethod
    def setUpClass(cls):
        cls._data_dir = tempfile.mkdtemp()
        cls.db = lmdb.open(os.path.join(cls._data_dir, 'db'))
        for i in xrange(2):
            if cls.SAME_SHAPE:
                width = 10
            else:
                width = 10 + i
            datum = cls.create_datum(10, width, 3)
            with cls.db.begin(write=True) as txn:
                txn.put(str(i), datum.SerializeToString())

    @classmethod
    def tearDownClass(cls):
        cls.db.close()
        shutil.rmtree(cls._data_dir)

    @staticmethod
    def create_datum(*shape):
        """
        Creates a datum with an image of the given shape
        """
        image = np.ones(shape, dtype='uint8')
        return caffe.io.array_to_datum(image)

    def test_defaults(self):
        assert analyze_db.analyze_db(self.db.path()) == self.PASS_DEFAULTS

    def test_force_shape(self):
        assert analyze_db.analyze_db(self.db.path(), force_same_shape=True) == self.PASS_FORCE


class TestSameShape(BaseTestWithDB):
    pass


class TestDifferentShape(BaseTestWithDB):
    SAME_SHAPE = False
    PASS_FORCE = False
