#!/usr/bin/env python2
# Copyright (c) 2015-2017, NVIDIA CORPORATION.  All rights reserved.

import argparse
from collections import Counter
import logging
import operator
import os.path
import sys
import time

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

import lmdb
import numpy as np
import PIL.Image

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import digits.config  # noqa
from digits import log  # noqa

# Import digits.config first to set path to Caffe
import caffe.io  # noqa
import caffe_pb2  # noqa

logger = logging.getLogger('digits.tools.analyze_db')
np.set_printoptions(suppress=True, precision=3)


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
                             map_size=1024**3,  # 1MB
                             readonly=True, lock=False)

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
               force_same_shape=False,
               print_data=False,
               ):
    """
    Looks at the data in a prebuilt database and verifies it
        Also prints out some information about it
    Returns True if all entries are valid

    Arguments:
    database -- path to the database

    Keyword arguments:
    only_count -- only count the entries, don't inspect them
    force_same_shape -- throw an error if not all images have the same shape
    print_data -- print the array for each datum
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

    unique_shapes = Counter()

    count = 0
    update_time = None
    for key, value in reader.entries():
        datum = caffe_pb2.Datum()
        datum.ParseFromString(value)

        if print_data:
            array = caffe.io.datum_to_array(datum)
            print '>>> Datum #%d (shape=%s)' % (count, array.shape)
            print array

        if (not datum.HasField('height') or datum.height == 0 or
                not datum.HasField('width') or datum.width == 0):
            if datum.encoded:
                if force_same_shape or not len(unique_shapes.keys()):
                    # Decode datum to learn the shape
                    s = StringIO()
                    s.write(datum.data)
                    s.seek(0)
                    img = PIL.Image.open(s)
                    width, height = img.size
                    channels = len(img.split())
                else:
                    # We've already decoded one image, don't bother reading the rest
                    width = '?'
                    height = '?'
                    channels = '?'
            else:
                errstr = 'Shape is not set and datum is not encoded'
                logger.error(errstr)
                raise ValueError(errstr)
        else:
            width, height, channels = datum.width, datum.height, datum.channels

        shape = '%sx%sx%s' % (width, height, channels)

        unique_shapes[shape] += 1

        if force_same_shape and len(unique_shapes.keys()) > 1:
            logger.error("Images with different shapes found: %s and %s" % tuple(unique_shapes.keys()))
            return False

        count += 1
        # Send update every 2 seconds
        if update_time is None or (time.time() - update_time) > 2:
            logger.debug('>>> Key %s' % key)
            print_datum(datum)
            logger.debug('Progress: %s/%s' % (count, reader.total_entries))
            update_time = time.time()

        if only_count:
            # quit after reading one
            count = reader.total_entries
            logger.info('Assuming all entries have same shape ...')
            unique_shapes[unique_shapes.keys()[0]] = count
            break

    if count != reader.total_entries:
        logger.warning('LMDB reported %s total entries, but only read %s' % (reader.total_entries, count))

    for key, val in sorted(unique_shapes.items(), key=operator.itemgetter(1), reverse=True):
        logger.info('%s entries found with shape %s (WxHxC)' % (val, key))

    logger.info('Completed in %s seconds.' % (time.time() - start_time,))
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze-Db tool - DIGITS')

    # Positional arguments

    parser.add_argument('database',
                        help='Path to the database')

    # Optional arguments

    parser.add_argument('--only-count',
                        action="store_true",
                        help="Only print the number of entries, don't analyze the data")
    parser.add_argument('--force-same-shape',
                        action="store_true",
                        help='Throw an error if not all entries have the same shape')
    parser.add_argument('--print-data',
                        action="store_true",
                        help='Print the array for each datum (best used with --only-count)')

    args = vars(parser.parse_args())

    if analyze_db(args['database'],
                  only_count=args['only_count'],
                  force_same_shape=args['force_same_shape'],
                  print_data=args['print_data'],
                  ):
        sys.exit(0)
    else:
        sys.exit(1)
