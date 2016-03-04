#!/usr/bin/env python2
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.

import argparse
import base64
import h5py
import logging
import numpy as np
import PIL.Image
import os
import sys

# Add path for DIGITS package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import digits.config
digits.config.load_config()
from digits import frameworks
from digits import utils, log
from digits.dataset import GenericImageDatasetJob
from digits.dataset import ImageClassificationDatasetJob
from digits.inference.errors import InferenceError
from digits.job import Job

# must call digits.config.load_config() before caffe to set the path
import caffe.io
import caffe_pb2

logger = logging.getLogger('digits.tools.inference')

"""
Perform inference on a list of images using the specified model
"""
def infer(input_list, output_dir, jobs_dir, model_id, epoch, batch_size, layers, gpu):

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
        raise InferenceError("Unable to find snapshot for epoch=%s" % repr(self.epoch))

    # retrieve image dimensions and resize mode
    if isinstance(dataset, ImageClassificationDatasetJob):
        height = dataset.image_dims[0]
        width = dataset.image_dims[1]
        channels = dataset.image_dims[2]
        resize_mode = dataset.resize_mode
    elif isinstance(dataset, GenericImageDatasetJob):
        db_task = dataset.analyze_db_tasks()[0]
        height = db_task.image_height
        width = db_task.image_width
        channels = db_task.image_channels
        resize_mode = 'squash'
    else:
        raise InferenceError("Unknown dataset type")

    n_input_samples = 0  # number of samples we were able to load
    input_ids = []       # indices of samples within file list
    input_data = []      # sample data

    # load paths from file
    paths = None
    with open(input_list) as infile:
        paths = infile.readlines()
    # load and resize images
    for idx, path in enumerate(paths):
        path = path.strip()
        try:
            image = utils.image.load_image(path.strip())
            image = utils.image.resize_image(image,
                        height, width,
                        channels    = channels,
                        resize_mode = resize_mode,
                        )
            input_ids.append(idx)
            input_data.append(image)
            n_input_samples = n_input_samples + 1
        except utils.errors.LoadImageError as e:
            print e

    # perform inference
    visualizations = None
    predictions = []

    if n_input_samples == 0:
        raise InferenceError("Unable to load any image from file '%s'" % repr(input_list))
    elif n_input_samples == 1:
        # single image inference
        outputs, visualizations = model.train_task().infer_one(input_data[0], snapshot_epoch=epoch, layers=layers, gpu=gpu)
    else:
        assert layers == 'none'
        outputs = model.train_task().infer_many(input_data, snapshot_epoch=epoch, gpu=gpu)

    # write to hdf5 file
    db_path = os.path.join(output_dir, 'inference.hdf5')
    db = h5py.File(db_path, 'w')

    # write input paths and images to database
    db.create_dataset("input_ids", data = input_ids)
    db.create_dataset("input_data", data = input_data)

    # write outputs to database
    db_outputs = db.create_group("outputs")
    for output_id, output_name in enumerate(outputs.keys()):
        output_data = outputs[output_name]
        output_key = base64.urlsafe_b64encode(str(output_name))
        dset = db_outputs.create_dataset(output_key, data=output_data)
        # add ID attribute so outputs can be sorted in
        # the order they appear in here
        dset.attrs['id'] = output_id

    # write visualization data
    if visualizations is not None and len(visualizations)>0:
        db_layers = db.create_group("layers")
        for idx, layer in enumerate(visualizations):
            vis = layer['vis'] if layer['vis'] is not None else np.empty(0)
            dset = db_layers.create_dataset(str(idx), data=vis)
            dset.attrs['name'] = layer['name']
            dset.attrs['vis_type'] = layer['vis_type']
            if 'param_count' in layer:
                dset.attrs['param_count'] = layer['param_count']
            if 'layer_type' in layer:
                dset.attrs['layer_type'] = layer['layer_type']
            dset.attrs['shape'] = layer['data_stats']['shape']
            dset.attrs['mean'] = layer['data_stats']['mean']
            dset.attrs['stddev'] = layer['data_stats']['stddev']
            dset.attrs['histogram_y'] = layer['data_stats']['histogram'][0]
            dset.attrs['histogram_x'] = layer['data_stats']['histogram'][1]
            dset.attrs['histogram_ticks'] = layer['data_stats']['histogram'][2]
    db.close()
    logger.info('Saved data to %s', db_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Inference tool - DIGITS')

    ### Positional arguments

    parser.add_argument('input_list',
            help='An input file containing paths to input data')
    parser.add_argument('output_dir',
            help='Directory to write outputs to')
    parser.add_argument('model',
            help='Model ID')

    ### Optional arguments
    parser.add_argument('-e', '--epoch',
            default='-1',
            help="Epoch (-1 for last)"
            )

    parser.add_argument('-j', '--jobs_dir',
            default='none',
            help='Jobs directory (default: from DIGITS config)',
            )

    parser.add_argument('-l', '--layers',
            default='none',
            help='Which layers to write to output ("none" [default] or "all")',
            )

    parser.add_argument('-b', '--batch_size',
            type=int,
            default=1,
            help='Batch size',
            )

    parser.add_argument('-g', '--gpu',
            type=int,
            default=None,
            help='GPU to use (as in nvidia-smi output, default: None)',
            )

    args = vars(parser.parse_args())

    try:
        infer(
            args['input_list'],
            args['output_dir'],
            args['jobs_dir'],
            args['model'],
            args['epoch'],
            args['batch_size'],
            args['layers'],
            args['gpu'],
                )
    except Exception as e:
        logger.error('%s: %s' % (type(e).__name__, e.message))
        raise

