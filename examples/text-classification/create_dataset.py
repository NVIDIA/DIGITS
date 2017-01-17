#!/usr/bin/env python2
# Copyright (c) 2016-2017, NVIDIA CORPORATION.  All rights reserved.
"""
Functions for creating a text classification dataset out of .csv files
The expected CSV structure is:
<Class>,<Text Field 1>, ..., <Text Field N>
"""

import argparse
import caffe
import csv
import lmdb
import numpy as np
import os
import PIL.Image
import shutil
import time

DB_BATCH_SIZE = 1024
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+ =<>()[]{}"
FEATURE_LEN = 1024  # must have integer square root


def _save_image(image, filename):
    # convert from (channels, heights, width) to (height, width)
    image = image[0]
    image = PIL.Image.fromarray(image)
    image.save(filename)


def create_dataset(folder, input_file_name, db_batch_size=None, create_images=False, labels_file=None):
    """
    Creates LMDB database and images (if create_images==True)
    """

    if db_batch_size is None:
        db_batch_size = DB_BATCH_SIZE

    # open output LMDB
    output_db = lmdb.open(folder, map_async=True, max_dbs=0)

    print "Reading input file %s..." % input_file_name
    # create character dict
    cdict = {}
    for i, c in enumerate(ALPHABET):
        cdict[c] = i + 2  # indices start at 1, skip first index for 'other' characters
    samples = {}
    with open(input_file_name) as f:
        reader = csv.DictReader(f, fieldnames=['class'], restkey='fields')
        for row in reader:
            label = row['class']
            if label not in samples:
                samples[label] = []
            sample = np.ones(FEATURE_LEN)  # one by default (i.e. 'other' character)
            count = 0
            for field in row['fields']:
                for char in field.lower():
                    if char in cdict:
                        sample[count] = cdict[char]
                    count += 1
                    if count >= FEATURE_LEN - 1:
                        break
            samples[label].append(sample)
        samples_per_class = None
        classes = samples.keys()
        class_samples = []
        for c in classes:
            if samples_per_class is None:
                samples_per_class = len(samples[c])
            else:
                assert samples_per_class == len(samples[c])
            class_samples.append(samples[c])

    indices = np.arange(samples_per_class)
    np.random.shuffle(indices)

    labels = None
    if labels_file is not None:
        labels = map(str.strip, open(labels_file, "r").readlines())
        assert len(classes) == len(samples)
    else:
        labels = classes
    print "Class labels: %s" % repr(labels)

    if create_images:
        for label in labels:
            os.makedirs(os.path.join(args['output'], label))

    print "Storing data into %s..." % folder

    batch = []
    for idx in indices:
        for c, cname in enumerate(classes):
            class_id = c + 1  # indices start at 1
            sample = class_samples[c][idx].astype('uint8')
            sample = sample[np.newaxis, np.newaxis, ...]
            sample = sample.reshape((1, np.sqrt(FEATURE_LEN), np.sqrt(FEATURE_LEN)))
            if create_images:
                filename = os.path.join(args['output'], labels[c], '%d.png' % idx)
                _save_image(sample, filename)
            datum = caffe.io.array_to_datum(sample, class_id)
            batch.append(('%d_%d' % (idx, class_id), datum))
        if len(batch) >= db_batch_size:
            _write_batch_to_lmdb(output_db, batch)
            batch = []

    # close database
    output_db.close()

    return


def _write_batch_to_lmdb(db, batch):
    """
    Write a batch of (key,value) to db
    """
    try:
        with db.begin(write=True) as lmdb_txn:
            for key, datum in batch:
                lmdb_txn.put(key, datum.SerializeToString())
    except lmdb.MapFullError:
        # double the map_size
        curr_limit = db.info()['map_size']
        new_limit = curr_limit * 2
        try:
            db.set_mapsize(new_limit)  # double it
        except AttributeError as e:
            version = tuple(int(x) for x in lmdb.__version__.split('.'))
            if version < (0, 87):
                raise ImportError('py-lmdb is out of date (%s vs 0.87)' % lmdb.__version__)
            else:
                raise e
        # try again
        _write_batch_to_lmdb(db, batch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Dataset tool')

    # Positional arguments
    parser.add_argument('input', help='Input .csv file')
    parser.add_argument('output', help='Output Folder')
    parser.add_argument('--create-images', action='store_true')
    parser.add_argument('--labels', default=None)

    args = vars(parser.parse_args())

    if os.path.exists(args['output']):
        shutil.rmtree(args['output'])

    os.makedirs(args['output'])

    start_time = time.time()

    create_dataset(
        args['output'],
        args['input'],
        create_images=args['create_images'],
        labels_file=args['labels'],
    )

    print 'Done after %s seconds' % (time.time() - start_time,)
