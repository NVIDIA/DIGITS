# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import random
import shutil
import tempfile

from nose.tools import assert_raises

from . import filesystem as fs
from digits import test_utils


test_utils.skipIfNotFramework('none')


class TestTreeSize():

    def test_bad_path(self):
        for path in [
                'some string',
                '/tmp/not-a-file',
                'http://not-a-url',
        ]:
            yield self.check_bad_path, path

    def check_bad_path(self, path):
        assert_raises(ValueError, fs.get_tree_size, path)

    def test_empty_folder(self):
        try:
            dir = tempfile.mkdtemp()
            assert(fs.get_tree_size(dir) == 0)
        finally:
            shutil.rmtree(dir)

    def test_folder_with_files(self):
        for n_files in [1, 5, 10]:
            yield self.check_folder_with_files, n_files

    def check_folder_with_files(self, n_files):
        try:
            dir = tempfile.mkdtemp()
            total_size = 0
            for i in range(n_files):
                # create file with random size of up to 1MB
                size = random.randint(1, 2**20)
                fd, name = tempfile.mkstemp(dir=dir)
                f = open(name, "w")
                f.seek(size - 1)
                f.write("\0")
                f.close()
                os.close(fd)
                total_size += size
            tree_size = fs.get_tree_size(dir)
            assert tree_size == total_size, "Expected size=%d, got %d" % (total_size, tree_size)
        finally:
            shutil.rmtree(dir)
