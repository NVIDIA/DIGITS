# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.

import shutil
import tempfile
import unittest

from . import create_generic_db
from digits import extensions
from digits import test_utils
from digits.utils import constants


test_utils.skipIfNotFramework('none')


class BaseTest(object):

    FEATURE_ENCODING = "png"
    LABEL_ENCODING = "none"
    BATCH_SIZE = 256
    NUM_THREADS = 2

    """
    Provides some helpful files and utilities
    """
    @classmethod
    def setUpClass(cls):
        if extensions.data.get_extension(cls.EXTENSION_ID) is None:
            raise unittest.SkipTest('Extension "%s" is not installed' % cls.EXTENSION_ID)
        cls.dataset_dir = tempfile.mkdtemp()
        cls.extension_class = extensions.data.get_extension(cls.EXTENSION_ID)
        cls.extension = cls.extension_class(**cls.EXTENSION_PARAMS)

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(cls.dataset_dir)
        except OSError:
            raise


class BaseTestGradientsExtension(BaseTest):
    """
    Create databases for the gradient extension
    """

    EXTENSION_ID = "image-gradients"
    EXTENSION_PARAMS = {
        "train_image_count": 10000,
        "val_image_count": 50,
        "test_image_count": 10,
        "image_width": 256,
        "image_height": 128
    }
    FORCE_SAME_SHAPE = True

    def create_db(self, stage):
        # create main DB creator object and execute main method
        db_creator = create_generic_db.DbCreator()
        db_creator.create_db(
            self.extension,
            stage,
            self.dataset_dir,
            self.BATCH_SIZE,
            self.NUM_THREADS,
            self.FEATURE_ENCODING,
            self.LABEL_ENCODING,
            force_same_shape=self.FORCE_SAME_SHAPE)

    def test_create_stages(self):
        for stage in (constants.TRAIN_DB, constants.VAL_DB, constants.TEST_DB):
            yield self.create_db, stage


class TestGradientsExtension(BaseTestGradientsExtension):
    FORCE_SAME_SHAPE = True


class TestGradientsExtensionDontForceSameShape(BaseTestGradientsExtension):
    FORCE_SAME_SHAPE = False
