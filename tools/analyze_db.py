#!/usr/bin/env python
# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

import sys
import os.path
import time
import argparse
import logging
import operator
from collections import Counter

try:
    import digits
except ImportError:
    # Add path for DIGITS package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config
digits.config.load_config()
from digits import utils, log

# must call digits.config.load_config() before caffe to set the path
try:
    import caffe_pb2
except ImportError:
    # See issue #32
    from caffe.proto import caffe_pb2
import lmdb

logger = logging.getLogger('digits.tools.analyze_db')

class DbReader(object):
    """
    Reads a database
    """

    def __init__(self, location):
        """
        Arguments:
        location -- where is the database
        """
        self._db = lmdb.open(location,
                map_size=1024**3, # 1MB
                max_dbs=0,
                readonly=True)

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


def validate_database_path(database):
    """
    Returns a valid database path
    Throws ValueErrors
    """
    p = os.path.abspath(database)
    if not os.path.exists(p):
        raise ValueError('No such file or directory')
    if os.path.isfile(p):
        p = os.path.dirname(p)
    if not os.path.isdir(p):
        raise ValueError('Not a directory')
    return p

def print_datum(datum):
    """
    Utility for printing a datum
    """
    logger.debug('\tWxHxC:   %sx%sx%s' % (datum.width, datum.height, datum.channels))
    logger.debug('\tLabel:   %s' % (datum.label if datum.HasField('label') else 'None',))
    logger.debug('\tEncoded: %s' % datum.encoded)


def analyze_db(database,
        only_count=False,
        force_dimensions=False,
        ):
    """
    Looks at the data in a prebuilt database and verifies it
        Also prints out some information about it
    Returns True if all entries are valid

    Arguments:
    database -- path to the database

    Keyword arguments:
    only_count -- only count the entries, don't inspect them
    force_dimensions -- throw an error if not all images have the same dimensions
    """
    start_time = time.time()

    # Open database
    try:
        database = validate_database_path(database)
    except ValueError as e:
        logger.error(e.message)
        return False

    reader = DbReader(database)
    logger.info('Total entries: %s' % reader.total_entries)

    if only_count:
        return True

    unique_dims = Counter()

    count = 0
    update_time = None
    for key, value in reader.entries():
        datum = caffe_pb2.Datum()
        datum.ParseFromString(value)

        d = '%sx%sx%s' % (datum.width, datum.height, datum.channels)
        unique_dims[d] += 1

        if force_dimensions and len(unique_dims.keys()) > 1:
            logger.error("Images with different sizes found: %s and %s" % tuple(sizes.keys()))
            return False

        count += 1
        # Send update every 2 seconds
        if update_time is None or (time.time() - update_time) > 2:
            logger.debug('>>> Key %s' % key)
            print_datum(datum)
            logger.debug('Progress: %s/%s' % (count, reader.total_entries))
            update_time = time.time()

    if count != reader.total_entries:
        logger.warning('LMDB reported %s total entries, but only read %s' % (reader.total_entries, count))

    for key, val in sorted(unique_dims.items(), key=operator.itemgetter(1), reverse=True):
        logger.info('%s entries found with dims %s (WxHxC)' % (val, key))

    logger.info('Completed in %s seconds.' % (time.time() - start_time,))
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze-Db tool - DIGITS')

    ### Positional arguments

    parser.add_argument('database',
            help='Path to the database')

    ### Optional arguments

    parser.add_argument('--only-count',
            action="store_true",
            help="Only print the number of entries, don't analyze the data")

    parser.add_argument('--force-dimensions',
            action="store_true",
            help="Throw an error if not all images have the same dimensions")

    args = vars(parser.parse_args())

    if analyze_db(args['database'],
            only_count = args['only_count'],
            force_dimensions = args['force_dimensions'],
            ):
        sys.exit(0)
    else:
        sys.exit(1)

