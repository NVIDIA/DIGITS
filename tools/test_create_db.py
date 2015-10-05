# Copyright (c) 2014-2015, NVIDIA CORPORATION.  All rights reserved.

import os.path
import tempfile
import shutil
from cStringIO import StringIO
import unittest
import platform
import Queue
from collections import Counter
import shutil

import nose.tools
import mock
import PIL.Image
import numpy as np

from . import create_db as _

class BaseTest():
    """
    Provides some helpful files and utilities
    """
    @classmethod
    def setUpClass(cls):
        cls.empty_file = tempfile.mkstemp()
        cls.empty_dir = tempfile.mkdtemp()

        # Create one good textfile
        cls.good_file = tempfile.mkstemp()

        # Create a color image
        cls.color_image_file = tempfile.mkstemp(suffix='.png')
        cls.numpy_image_color = np.ones((8,10,3), dtype='uint8')
        cls.pil_image_color = PIL.Image.fromarray(cls.numpy_image_color)
        cls.pil_image_color.save(cls.color_image_file[1])

        # Create a grayscale image
        cls.gray_image_file = tempfile.mkstemp(suffix='.png')
        cls.numpy_image_gray = np.ones((8,10), dtype='uint8')
        cls.pil_image_gray = PIL.Image.fromarray(cls.numpy_image_gray)
        cls.pil_image_gray.save(cls.gray_image_file[1])

        cls.image_count = 0
        for i in xrange(3):
            for j in xrange(3):
                os.write(cls.good_file[0], '%s %s\n' % (cls.color_image_file[1], i))
                os.write(cls.good_file[0], '%s %s\n' % (cls.gray_image_file[1], i))
                cls.image_count += 2

    @classmethod
    def tearDownClass(cls):
        for f in cls.empty_file, cls.good_file, cls.color_image_file, cls.gray_image_file:
            try:
                os.close(f[0])
                os.remove(f[1])
            except OSError:
                pass
        try:
            shutil.rmtree(cls.empty_dir)
        except OSError:
            raise


class TestFillLoadQueue(BaseTest):

    def test_valid_file(self):
        for shuffle in True, False:
            yield self.check_valid_file, shuffle

    def check_valid_file(self, shuffle):
        queue = Queue.Queue()
        result = _._fill_load_queue(self.good_file[1], queue, shuffle)
        assert result == self.image_count, 'lines not added'
        assert queue.qsize() == self.image_count, 'queue not full'

    def test_empty_file(self):
        for shuffle in True, False:
            yield self.check_empty_file, shuffle

    def check_empty_file(self, shuffle):
        queue = Queue.Queue()
        nose.tools.assert_raises(
                _.BadInputFileError,
                _._fill_load_queue,
                self.empty_file[1], queue, shuffle)

class TestParseLine():

    def test_good_lines(self):
        for label, line in [
                (0, '/path/image.jpg 0'),
                (1, 'image.jpg 1'),
                (2, 'image.jpg 2\n'),
                (3, 'image.jpg           3'),
                (4, 'spaces in filename.jpg 4'),
                ]:
            yield self.check_good_line, line, label

    def check_good_line(self, line, label):
        c = Counter()
        p, l = _._parse_line(line, c)
        assert l == label, 'parsed label wrong'
        assert c[l] == 1, 'distribution is wrong'

    def test_bad_lines(self):
        for line in [
                'nolabel.jpg',
                'non-number.jpg five',
                'negative.jpg -1',
                ]:
            yield self.check_bad_line, line

    def check_bad_line(self, line):
        nose.tools.assert_raises(
                _.ParseLineError,
                _._parse_line,
                line, Counter()
                )


class TestCalculateBatchSize():
    def test(self):
        for count, batch_size in [
                (1, 1),
                (50, 50),
                (100, 100),
                (200, 100),
                ]:
            yield self.check, count, batch_size

    def check(self, count, batch_size):
        assert _._calculate_batch_size(count) == batch_size


class TestCalculateNumThreads():
    def test(self):
        for batch_size, shuffle, num in [
                (1000, True, 10),
                (1000, False, 1),
                (100, True, 10),
                (100, False, 1),
                (50, True, 7),
                (4, True, 2),
                (1, True, 1),
                ]:
            yield self.check, batch_size, shuffle, num

    def check(self, batch_size, shuffle, num):
        assert _._calculate_num_threads(
                batch_size, shuffle) == num


class TestInitialImageSum():
    def test_color(self):
        s = _._initial_image_sum(10, 10, 3)
        assert s.shape == (10, 10, 3)
        assert s.dtype == 'float64'

    def test_grayscale(self):
        s = _._initial_image_sum(10, 10, 1)
        assert s.shape == (10, 10)
        assert s.dtype == 'float64'


class TestImageToDatum(BaseTest):
    def test(self):
        for compression in None, 'png', 'jpg':
            yield self.check_color, compression
            yield self.check_grayscale, compression

    def check_color(self, compression):
        d = _._array_to_datum(self.numpy_image_color, 1, compression)
        assert d.height == self.numpy_image_color.shape[0]
        assert d.width == self.numpy_image_color.shape[1]
        assert d.channels == 3
        assert d.encoded == bool(compression)

    def check_grayscale(self, compression):
        d = _._array_to_datum(self.numpy_image_gray, 1, compression)
        assert d.height == self.numpy_image_gray.shape[0]
        assert d.width == self.numpy_image_gray.shape[1]
        assert d.channels == 1
        assert d.encoded == bool(compression)

class TestSaveMeans():
    def test(self):
        for color in True, False:
            d = tempfile.mkdtemp()
            for filename in 'mean.jpg', 'mean.png', 'mean.npy', 'mean.binaryproto':
                yield self.check, d, filename, color
            shutil.rmtree(d)

    def check(self, directory, filename, color):
        filename = os.path.join(directory, filename)
        if color:
            s = np.ones((8,10,3),dtype='float64')
        else:
            s = np.ones((8,10),dtype='float64')

        _._save_means(s, 2, [filename])
        assert os.path.exists(filename)

class BaseCreationTest(BaseTest):

    def test_image_sizes(self):
        for width in 8, 12:
            for channels in 1, 3:
                yield self.check_image_sizes, width, channels, False

    def check_image_sizes(self, width, channels, shuffle):
        _.create_db(self.good_file[1], os.path.join(self.empty_dir, 'db'),
                width, 10, channels, self.BACKEND)

    def test_no_shuffle(self):
        _.create_db(self.good_file[1], os.path.join(self.empty_dir, 'db'),
                10, 10, 1, self.BACKEND, shuffle=False)

    def test_means(self):
        mean_files = []
        for suffix in 'jpg','npy','png','binaryproto':
            mean_files.append(os.path.join(self.empty_dir, 'mean.%s' % suffix))
        _.create_db(self.good_file[1], os.path.join(self.empty_dir, 'db'),
                10, 10, 1, self.BACKEND, mean_files=mean_files)

    def test_augmentation_rotation(self):
        _.create_db(self.good_file[1], os.path.join(self.empty_dir, 'db'),
                10, 10, 1, self.BACKEND, augmentation_rotation=True, 
                augmentation_rotation_probability=0.75,
                augmentation_rotation_angle_min=-90,
                augmentation_rotation_angle_max=90)

    def test_augmentation_rotation_color(self):
        _.create_db(self.good_file[1], os.path.join(self.empty_dir, 'db'),
                10, 10, 3, self.BACKEND, augmentation_rotation=True, 
                augmentation_rotation_probability=0.75,
                augmentation_rotation_angle_min=-90,
                augmentation_rotation_angle_max=90)

    def test_augmentation_contrast(self):
        _.create_db(self.good_file[1], os.path.join(self.empty_dir, 'db'),
                10, 10, 1, self.BACKEND, augmentation_contrast=True, 
                augmentation_contrast_probability=0.75,
                augmentation_contrast_strength_min=0.75,
                augmentation_contrast_strength_max=1.25)

    def test_augmentation_hue(self):
        _.create_db(self.good_file[1], os.path.join(self.empty_dir, 'db'),
                10, 10, 3, self.BACKEND, augmentation_hue=True, 
                augmentation_hue_probability=0.75,
                augmentation_hue_angle_min=66,
                augmentation_hue_angle_max=122)

    def test_augmentation_translation(self):
        _.create_db(self.good_file[1], os.path.join(self.empty_dir, 'db'),
                10, 10, 3, self.BACKEND, augmentation_translation=True, 
                augmentation_translation_probability=0.75,
                augmentation_translation_dx_min=-0.5,
                augmentation_translation_dx_max=0.5,
                augmentation_translation_dy_min=-0.5,
                augmentation_translation_dy_max=0.5)
 

class TestLmdbCreation(BaseCreationTest):
    BACKEND = 'lmdb'

class TestHdf5Creation(BaseCreationTest):
    BACKEND = 'hdf5'

    def test_dset_limit(self):
        db_dir = os.path.join(self.empty_dir, 'db')
        _.create_db(self.good_file[1], db_dir,
                10, 10, 1, 'hdf5', hdf5_dset_limit=10*10)
        with open(os.path.join(db_dir, 'list.txt')) as infile:
            lines = infile.readlines()
            assert len(lines) == self.image_count, '%d != %d' % (len(lines), self.image_count)

