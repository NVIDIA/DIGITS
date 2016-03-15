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
Perform inference on a batch_size
"""
def infer_batch(model, input_data, epoch, layers, gpu):
    if layers != 'none':
        assert(len(input_data) == 1)
        # single image inference
        return model.train_task().infer_one(input_data[0], snapshot_epoch=epoch, layers=layers, gpu=gpu)
    else:
        return model.train_task().infer_many(input_data, snapshot_epoch=epoch, gpu=gpu), None

"""
Save visualizations to database
"""
def save_visualizations(db, visualizations):
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

"""
Save chunk to db group
"""
def save_chunk(grp, name, max_samples, batch_len, idx, data, attributes = {}):
    assert(data.shape[0] == batch_len)
    if name not in grp:
        dset = grp.create_dataset(name, data = data, maxshape = (max_samples,) + data[0].shape)
        for key, value in attributes.items():
            dset.attrs[key] = value
    else:
        grp[name].resize((idx,) + data[0].shape)
        grp[name][(idx - batch_len):] = data

"""
Save data
"""
def save_data(db, max_samples, idx, input_ids, input_data, outputs):
    batch_len = len(input_ids)
    # write input data
    save_chunk(db, "input_ids", max_samples, batch_len, idx, np.array(input_ids))
    save_chunk(db, "input_data", max_samples, batch_len, idx, np.array(input_data))
    # write output data
    grp = db["outputs"]
    for output_id, output_name in enumerate(outputs.keys()):
        output_data = outputs[output_name]
        output_key = base64.urlsafe_b64encode(str(output_name))
        output_key = base64.urlsafe_b64encode(str(output_name))
        save_chunk(grp, output_key, max_samples, batch_len, idx, output_data, {'id': output_id})

"""
Perform inference on a list of images using the specified model
and save inference data to an HDF5 db
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

    # retrieve batch size (unless specified on command line)
    if batch_size is None:
        batch_size = task.get_test_batch_size()

    n_loaded_samples = 0  # number of samples we were able to load
    input_ids = []       # indices of samples within file list
    input_data = []      # sample data

    # create hdf5 file
    db_path = os.path.join(output_dir, 'inference.hdf5')
    db = h5py.File(db_path, 'w')
    db.create_group("outputs")

    # load paths from file
    paths = None
    with open(input_list) as infile:
        paths = infile.readlines()
    n_input_paths = len(paths)

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
            n_loaded_samples = n_loaded_samples + 1
        except utils.errors.LoadImageError as e:
            print e
        # do we have a full batch, or have we reached the last item?
        if (not n_loaded_samples % batch_size) or (idx == n_input_paths - 1):
            # any item(s) left to save?
            if len(input_ids) > 0:
                # perform inference
                outputs, visualizations = infer_batch(model, input_data, epoch, layers, gpu)
                # save visualizations
                if visualizations is not None and len(visualizations)>0:
                    save_visualizations(db, visualizations)
                # save other data
                save_data(db, n_input_paths, n_loaded_samples, input_ids, input_data, outputs)
                # empty input lists
                input_ids = []
                input_data = []
                logger.info('Processed %d/%d images', idx+1, n_input_paths)

    if n_loaded_samples == 0:
        raise InferenceError("Unable to load any image from file '%s'" % repr(input_list))

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
            default=None,
            help='Batch size (default: network\'s default)',
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

