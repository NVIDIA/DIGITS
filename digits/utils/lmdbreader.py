# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import lmdb


class DbReader(object):
    """
    Reads a database
    """

    def __init__(self, location):
        """
        Arguments:
        location -- where is the database
        """
        self._db = lmdb.open(
            location,
            map_size=1024**3,  # 1MB
            readonly=True,
            lock=False)

        with self._db.begin() as txn:
            self.total_entries = txn.stat()['entries']

    def entries(self):
        """
        Generator returning all entries in the DB
        """
        with self._db.begin() as txn:
            cursor = txn.cursor()
            for item in cursor:
                yield item
