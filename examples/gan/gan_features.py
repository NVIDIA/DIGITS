#!/usr/bin/env python2
# Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.

import argparse
import logging
import numpy as np
import pickle
import PIL.Image
import os
import sys
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import digits.config  # noqa
from digits import utils, log  # noqa
from digits.inference.errors import InferenceError  # noqa
from digits.job import Job  # noqa
from digits.utils.lmdbreader import DbReader  # noqa

# Import digits.config before caffe to set the path
import caffe_pb2  # noqa

logger = logging.getLogger('digits.tools.inference')

# number of image embeddings to store
N_EMBEDDINGS = 10000


def parse_datum(value):
    """
    Parse a Caffe datum
    """
    datum = caffe_pb2.Datum()
    datum.ParseFromString(value)
    if datum.encoded:
        s = StringIO()
        s.write(datum.data)
        s.seek(0)
        img = PIL.Image.open(s)
        img = np.array(img)
    else:
        import caffe.io
        arr = caffe.io.datum_to_array(datum)
        # CHW -> HWC
        arr = arr.transpose((1, 2, 0))
        if arr.shape[2] == 1:
            # HWC -> HW
            arr = arr[:, :, 0]
        elif arr.shape[2] == 3:
            # BGR -> RGB
            # XXX see issue #59
            arr = arr[:, :, [2, 1, 0]]
        img = arr
    return img


def save_attributes(attributes):
    """
    Save attribute vectors
    """
    zs = np.zeros(attributes['positive_attribute_z'].shape)
    for i in xrange(attributes['n_attributes']):
        zs[i] = attributes['positive_attribute_z'][i] / attributes['positive_count'][i] \
            - attributes['negative_attribute_z'][i] / attributes['negative_count'][i]
    output = open('attributes_z.pkl', 'wb')
    pickle.dump(zs, output)


def save_embeddings(embeddings):
    filename = 'embeddings.pkl'
    logger.info('Saving embeddings to %s...' % filename)
    output = open(filename, 'wb')
    pickle.dump(embeddings, output)


def infer(jobs_dir,
          model_id,
          epoch,
          batch_size,
          gpu):
    """
    Perform inference on a list of images using the specified model
    """
    # job directory defaults to that defined in DIGITS config
    if jobs_dir == 'none':
        jobs_dir = digits.config.config_value('jobs_dir')

    # load model job
    model_dir = os.path.join(jobs_dir, model_id)
    assert os.path.isdir(model_dir), "Model dir %s does not exist" % model_dir
    model = Job.load(model_dir)

    # load dataset job
    dataset_dir = os.path.join(jobs_dir, model.dataset_id)
    assert os.path.isdir(dataset_dir), "Dataset dir %s does not exist" % dataset_dir
    dataset = Job.load(dataset_dir)
    for task in model.tasks:
        task.dataset = dataset

    # retrieve snapshot file
    task = model.train_task()
    snapshot_filename = None
    epoch = float(epoch)
    if epoch == -1 and len(task.snapshots):
        # use last epoch
        epoch = task.snapshots[-1][1]
        snapshot_filename = task.snapshots[-1][0]
    else:
        for f, e in task.snapshots:
            if e == epoch:
                snapshot_filename = f
                break
    if not snapshot_filename:
        raise InferenceError("Unable to find snapshot for epoch=%s" % repr(epoch))

    input_data = []      # sample data
    input_labels = []    # sample labels

    # load images from database
    feature_db_path = dataset.get_feature_db_path(utils.constants.TRAIN_DB)
    feature_reader = DbReader(feature_db_path)

    label_db_path = dataset.get_label_db_path(utils.constants.TRAIN_DB)
    label_reader = DbReader(label_db_path)

    embeddings = {'count': 0, 'images': None, 'zs': None}

    def aggregate(images, labels, attributes, embeddings):
        # perform inference
        outputs = model.train_task().infer_many(
            images,
            snapshot_epoch=epoch,
            gpu=gpu,
            resize=False)
        z_vectors = outputs['output'][:, :100]
        for image, label, z in zip(images, labels, z_vectors):
            if embeddings['images'] is None:
                embeddings['images'] = np.empty((N_EMBEDDINGS,) + image.shape)
            if embeddings['zs'] is None:
                embeddings['zs'] = np.empty((N_EMBEDDINGS,) + z.shape)
            if embeddings['count'] < N_EMBEDDINGS:
                embeddings['images'][embeddings['count']] = image
                embeddings['zs'][embeddings['count']] = z
                embeddings['count'] += 1
                if embeddings['count'] == N_EMBEDDINGS:
                    save_embeddings(embeddings)

            for attribute in range(attributes['n_attributes']):
                if label[attribute] > 0:
                    attributes['positive_attribute_z'][attribute] += z
                    attributes['positive_count'][attribute] += 1
                else:
                    attributes['negative_attribute_z'][attribute] += z
                    attributes['negative_count'][attribute] += 1
        # save
        save_attributes(attributes)

    n_input_samples = 0
    label_len = None
    z_dim = 100
    for key, value in feature_reader.entries():
        img = parse_datum(value)
        label = parse_datum(label_reader.entry(key))[0]
        if label_len is None:
            label_len = len(label)
            attributes = {
                'n_attributes': label_len,
                'negative_count': np.zeros(label_len),
                'positive_count': np.zeros(label_len),
                'negative_attribute_z': np.zeros((label_len, z_dim)),
                'positive_attribute_z': np.zeros((label_len, z_dim)),
            }
        elif label_len != len(label):
            raise ValueError("label len differs: %d vs %d" % (label_len, len(label)))
        input_data.append(img)
        input_labels.append(label)
        n_input_samples = n_input_samples + 1
        if n_input_samples % batch_size == 0:
            aggregate(input_data, input_labels, attributes, embeddings)
            print("######## %d processed ########" % n_input_samples)
            input_data = []      # sample data
            input_labels = []    # sample labels

    if n_input_samples % batch_size != 0:
        aggregate(input_data, input_labels, attributes, embeddings)
        print("######## %d processed ########" % n_input_samples)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference tool - DIGITS')

    # Positional arguments

    parser.add_argument(
        'model',
        help='Model ID')

    # Optional arguments
    parser.add_argument(
        '-e',
        '--epoch',
        default='-1',
        help="Epoch (-1 for last)"
    )

    parser.add_argument(
        '-j',
        '--jobs_dir',
        default='none',
        help='Jobs directory (default: from DIGITS config)',
    )

    parser.add_argument(
        '-b',
        '--batch_size',
        type=int,
        default=1024,
        help='Batch size',
    )

    parser.add_argument(
        '-g',
        '--gpu',
        type=int,
        default=None,
        help='GPU to use (as in nvidia-smi output, default: None)',
    )

    parser.set_defaults(resize=True)

    args = vars(parser.parse_args())

    try:
        infer(
            args['jobs_dir'],
            args['model'],
            args['epoch'],
            args['batch_size'],
            args['gpu'],
        )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise
