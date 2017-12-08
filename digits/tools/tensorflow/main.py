#!/usr/bin/env python2
# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
#
# This document should comply with PEP-8 Style Guide
# Linter: pylint

"""
TensorFlow training executable for DIGITS
Defines the training procedure

Usage:
See the self-documenting flags below.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import datetime
import inspect
import json
import logging
import math
import numpy as np
import os
from six.moves import xrange  # noqa
import tensorflow as tf
import tensorflow.contrib.slim as slim  # noqa
from tensorflow.python.client import timeline, device_lib  # noqa
from tensorflow.python.ops import template  # noqa
from tensorflow.python.lib.io import file_io
from tensorflow.core.framework import summary_pb2
from tensorflow.python.tools import freeze_graph

# Local imports
import utils as digits
import lr_policy
from model import Model, Tower  # noqa
from utils import model_property  # noqa

import tf_data

# Constants
TF_INTRA_OP_THREADS = 0
TF_INTER_OP_THREADS = 0
MIN_LOGS_PER_TRAIN_EPOCH = 8  # torch default: 8

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

# Basic model parameters. #float, integer, boolean, string
tf.app.flags.DEFINE_integer('batch_size', 16, """Number of images to process in a batch""")
tf.app.flags.DEFINE_integer(
    'croplen', 0, """Crop (x and y). A zero value means no cropping will be applied""")
tf.app.flags.DEFINE_integer('epoch', 1, """Number of epochs to train, -1 for unbounded""")
tf.app.flags.DEFINE_string('inference_db', '', """Directory with inference file source""")
tf.app.flags.DEFINE_integer(
    'validation_interval', 1, """Number of train epochs to complete, to perform one validation""")
tf.app.flags.DEFINE_string('labels_list', '', """Text file listing label definitions""")
tf.app.flags.DEFINE_string('mean', '', """Mean image file""")
tf.app.flags.DEFINE_float('momentum', '0.9', """Momentum""")  # Not used by DIGITS front-end
tf.app.flags.DEFINE_string('network', '', """File containing network (model)""")
tf.app.flags.DEFINE_string('networkDirectory', '', """Directory in which network exists""")
tf.app.flags.DEFINE_string('optimization', 'sgd', """Optimization method""")
tf.app.flags.DEFINE_string('save', 'results', """Save directory""")
tf.app.flags.DEFINE_integer('seed', 0, """Fixed input seed for repeatable experiments""")
tf.app.flags.DEFINE_boolean('shuffle', False, """Shuffle records before training""")
tf.app.flags.DEFINE_float(
    'snapshotInterval', 1.0,
    """Specifies the training epochs to be completed before taking a snapshot""")
tf.app.flags.DEFINE_string('snapshotPrefix', '', """Prefix of the weights/snapshots""")
tf.app.flags.DEFINE_string(
    'subtractMean', 'none',
    """Select mean subtraction method. Possible values are 'image', 'pixel' or 'none'""")
tf.app.flags.DEFINE_string('train_db', '', """Directory with training file source""")
tf.app.flags.DEFINE_string(
    'train_labels', '',
    """Directory with an optional and seperate labels file source for training""")
tf.app.flags.DEFINE_string('validation_db', '', """Directory with validation file source""")
tf.app.flags.DEFINE_string(
    'validation_labels', '',
    """Directory with an optional and seperate labels file source for validation""")
tf.app.flags.DEFINE_string(
    'visualizeModelPath', '', """Constructs the current model for visualization""")
tf.app.flags.DEFINE_boolean(
    'visualize_inf', False, """Will output weights and activations for an inference job.""")
tf.app.flags.DEFINE_string(
    'weights', '', """Filename for weights of a model to use for fine-tuning""")

# @TODO(tzaman): is the bitdepth in line with the DIGITS team?
tf.app.flags.DEFINE_integer('bitdepth', 8, """Specifies an image's bitdepth""")

# @TODO(tzaman); remove torch mentions below
tf.app.flags.DEFINE_float('lr_base_rate', '0.01', """Learning rate""")
tf.app.flags.DEFINE_string(
    'lr_policy', 'fixed',
    """Learning rate policy. (fixed, step, exp, inv, multistep, poly, sigmoid)""")
tf.app.flags.DEFINE_float(
    'lr_gamma', -1,
    """Required to calculate learning rate. Applies to: (step, exp, inv, multistep, sigmoid)""")
tf.app.flags.DEFINE_float(
    'lr_power', float('Inf'),
    """Required to calculate learning rate. Applies to: (inv, poly)""")
tf.app.flags.DEFINE_string(
    'lr_stepvalues', '',
    """Required to calculate stepsize of the learning rate. Applies to: (step, multistep, sigmoid).
    For the 'multistep' lr_policy you can input multiple values seperated by commas""")

# Tensorflow-unique arguments for DIGITS
tf.app.flags.DEFINE_string(
    'save_vars', 'all',
    """Sets the collection of variables to be saved: 'all' or only 'trainable'.""")
tf.app.flags.DEFINE_string('summaries_dir', '', """Directory of Tensorboard Summaries (logdir)""")
tf.app.flags.DEFINE_boolean(
    'serving_export', False, """Flag for exporting an Tensorflow Serving model""")
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
tf.app.flags.DEFINE_integer(
    'log_runtime_stats_per_step', 0,
    """Logs runtime statistics for Tensorboard every x steps, defaults to 0 (off).""")

# Augmentation
tf.app.flags.DEFINE_string(
    'augFlip', 'none',
    """The flip options {none, fliplr, flipud, fliplrud} as randompre-processing augmentation""")
tf.app.flags.DEFINE_float(
    'augNoise', 0., """The stddev of Noise in AWGN as pre-processing augmentation""")
tf.app.flags.DEFINE_float(
    'augContrast', 0., """The contrast factor's bounds as sampled from a random-uniform distribution
     as pre-processing  augmentation""")
tf.app.flags.DEFINE_bool(
    'augWhitening', False, """Performs per-image whitening by subtracting off its own mean and
    dividing by its own standard deviation.""")
tf.app.flags.DEFINE_float(
    'augHSVh', 0., """The stddev of HSV's Hue shift as pre-processing  augmentation""")
tf.app.flags.DEFINE_float(
    'augHSVs', 0., """The stddev of HSV's Saturation shift as pre-processing  augmentation""")
tf.app.flags.DEFINE_float(
    'augHSVv', 0., """The stddev of HSV's Value shift as pre-processing augmentation""")


def save_timeline_trace(run_metadata, save_dir, step):
    tl = timeline.Timeline(run_metadata.step_stats)
    ctf = tl.generate_chrome_trace_format(show_memory=True)
    tl_fn = os.path.join(save_dir, 'timeline_%s.json' % step)
    with open(tl_fn, 'w') as f:
        f.write(ctf)
    logging.info('Timeline trace written to %s', tl_fn)


def strip_data_from_graph_def(graph_def):
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            if (tensor.tensor_content):
                tensor.tensor_content = ''
            if (tensor.string_val):
                del tensor.string_val[:]
    return strip_def


def visualize_graph(graph_def, path):
    graph_def = strip_data_from_graph_def(graph_def)
    logging.info('Writing Graph Definition..')
    file_io.write_string_to_file(path, str(graph_def))
    logging.info('Graph Definition Written.')


def average_head_keys(tags, vals):
    """ Averages keys with same end (head) name.
    Example: foo1/bar=1 and foo2/bar=2 should collapse to bar=1.5
    """
    tail_tags = [w.split('/')[-1] for w in tags]
    sums = {}
    nums = {}
    for a, b in zip(tail_tags, vals):
        if a not in sums:
            sums[a] = b
            nums[a] = 1
        else:
            sums[a] += b
            nums[a] += 1
    tags_clean = sums.keys()
    return tags_clean, np.asarray(sums.values())/np.asarray(nums.values())


def summary_to_lists(summary_str):
    """ Takes a Tensorflow stringified Summary object and returns only
    the scalar values to a list of tags and a list of values
    Args:
        summary_str: string of a Tensorflow Summary object
    Returns:
        tags: list of tags
        vals: list of values corresponding to the tag list

    """
    summ = summary_pb2.Summary()
    summ.ParseFromString(summary_str)
    tags = []
    vals = []
    for s in summ.value:
        if s.HasField('simple_value'):  # and s.simple_value: # Only parse scalar_summaries
            if s.simple_value == float('Inf') or np.isnan(s.simple_value):
                raise ValueError('Model diverged with %s = %s : Try decreasing your learning rate' %
                                 (s.tag, s.simple_value))
            tags.append(s.tag)
            vals.append(s.simple_value)
    tags, vals = average_head_keys(tags, vals)
    vals = np.asarray(vals)
    return tags, vals


def print_summarylist(tags, vals):
    """ Prints a nice one-line listing of tags and their values in a nice format
    that corresponds to how the DIGITS regex reads it.
    Args:
        tags: an array of tags
        vals: an array of values
    Returns:
        print_list: a string containing formatted tags and values
    """
    print_list = ''
    for i, key in enumerate(tags):
        if vals[i] == float('Inf'):
            raise ValueError('Infinite value %s = Inf' % key)
        print_list = print_list + key + " = " + "{:.6f}".format(vals[i])
        if i < len(tags)-1:
            print_list = print_list + ", "
    return print_list


def dump(obj):
    for attr in dir(obj):
        print("obj.%s = %s" % (attr, getattr(obj, attr)))


def load_snapshot(sess, weight_path, var_candidates):
    """ Loads a snapshot into a session from a weight path. Will only load the
    weights that are both in the weight_path file and the passed var_candidates."""
    logging.info("Loading weights from pretrained model - %s ", weight_path)
    reader = tf.train.NewCheckpointReader(weight_path)
    var_map = reader.get_variable_to_shape_map()

    # Only obtain all the variables that are [in the current graph] AND [in the checkpoint]
    vars_restore = []
    for vt in var_candidates:
        for vm in var_map.keys():
            if vt.name.split(':')[0] == vm:
                if ("global_step" not in vt.name) and not (vt.name.startswith("train/")):
                    vars_restore.append(vt)
                    logging.info('restoring %s -> %s' % (vm, vt.name))
                else:
                    logging.info('NOT restoring %s -> %s' % (vm, vt.name))

    logging.info('Restoring %s variable ops.' % len(vars_restore))
    tf.train.Saver(vars_restore, max_to_keep=0, sharded=FLAGS.serving_export).restore(sess, weight_path)
    logging.info('Variables restored.')


def save_snapshot(sess, saver, save_dir, snapshot_prefix, epoch, for_serving=False):
    """
    Saves a snapshot of the current session, saving all variables previously defined
    in the ctor of the saver. Also saves the flow of the graph itself (only once).
    """
    number_dec = str(FLAGS.snapshotInterval-int(FLAGS.snapshotInterval))[2:]
    if number_dec is '':
        number_dec = '0'
    epoch_fmt = "{:." + number_dec + "f}"

    snapshot_file = os.path.join(save_dir, snapshot_prefix + '_' + epoch_fmt.format(epoch) + '.ckpt')

    logging.info('Snapshotting to %s', snapshot_file)
    checkpoint_path = saver.save(sess, snapshot_file)
    logging.info('Snapshot saved.')

    if for_serving:
        # @TODO(tzaman) : we could further extend this by supporting tensorflow-serve
        logging.error('NotImplementedError: Tensorflow-Serving support.')
        exit(-1)

    # Past this point the graph shouldn't be changed, so saving it once is enough
    filename_graph = os.path.join(save_dir, snapshot_prefix + '.graph_def')
    if not os.path.isfile(filename_graph):
        with open(filename_graph, 'wb') as f:
            logging.info('Saving graph to %s', filename_graph)
            f.write(sess.graph_def.SerializeToString())
            logging.info('Saved graph to %s', filename_graph)
        # meta_graph_def = tf.train.export_meta_graph(filename='?')

    return checkpoint_path, filename_graph


def save_weight_visualization(w_names, a_names, w, a):
    try:
        import h5py
    except ImportError:
        logging.error("Attempt to create HDF5 Loader but h5py is not installed.")
        exit(-1)
    fn = os.path.join(FLAGS.save, 'vis.h5')
    vis_db = h5py.File(fn, 'w')
    db_layers = vis_db.create_group("layers")

    logging.info('Saving visualization to %s', fn)
    for i in range(0, len(w)):
        dset = db_layers.create_group(str(i))
        dset.attrs['var'] = w_names[i].name
        dset.attrs['op'] = a_names[i]
        if w[i].shape:
            dset.create_dataset('weights', data=w[i])
        if a[i].shape:
            dset.create_dataset('activations', data=a[i])
    vis_db.close()


def Inference(sess, model):
    """
    Runs one inference (evaluation) epoch (all the files in the loader)
    """

    inference_op = model.towers[0].inference
    if FLAGS.labels_list:  # Classification -> assume softmax usage
        # Append a softmax op
        inference_op = tf.nn.softmax(inference_op)

    weight_vars = []
    activation_ops = []
    if FLAGS.visualize_inf:
        trainable_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # Retrace the origin op of each variable
        for n in tf.get_default_graph().as_graph_def().node:
            for tw in trainable_weights:
                tw_name_reader = tw.name.split(':')[0] + '/read'
                if tw_name_reader in n.input:
                    node_op_name = n.name + ':0'  # @TODO(tzaman) this assumes exactly 1 output - allow to be dynamic!
                    weight_vars.append(tw)
                    activation_ops.append(node_op_name)
                    continue

    try:
        while not model.queue_coord.should_stop():
            keys, preds, [w], [a] = sess.run([model.dataloader.batch_k,
                                              inference_op,
                                              [weight_vars],
                                              [activation_ops]])

            if FLAGS.visualize_inf:
                save_weight_visualization(weight_vars, activation_ops, w, a)

            # @TODO(tzaman): error on no output?
            for i in range(len(keys)):
                #    for j in range(len(preds)):
                # We're allowing multiple predictions per image here. DIGITS doesnt support that iirc
                logging.info('Predictions for image ' + str(model.dataloader.get_key_index(keys[i])) +
                             ': ' + json.dumps(preds[i].tolist()))
    except tf.errors.OutOfRangeError:
        print('Done: tf.errors.OutOfRangeError')


def Validation(sess, model, current_epoch):
    """
    Runs one validation epoch.
    """

    # @TODO(tzaman): utilize the coordinator by resetting the queue after 1 epoch.
    # see https://github.com/tensorflow/tensorflow/issues/4535#issuecomment-248990633

    print_vals_sum = 0
    steps = 0
    while (steps * model.dataloader.batch_size) < model.dataloader.get_total():
        summary_str = sess.run(model.summary)
        # Parse the summary
        tags, print_vals = summary_to_lists(summary_str)
        print_vals_sum = print_vals + print_vals_sum
        steps += 1

    print_list = print_summarylist(tags, print_vals_sum/steps)

    logging.info("Validation (epoch " + str(current_epoch) + "): " + print_list)


def loadLabels(filename):
    with open(filename) as f:
        return f.readlines()


def main(_):

    # Always keep the cpu as default
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        if FLAGS.validation_interval == 0:
            FLAGS.validation_db = None

        # Set Tensorboard log directory
        if FLAGS.summaries_dir:
            # The following gives a nice but unrobust timestamp
            FLAGS.summaries_dir = os.path.join(FLAGS.summaries_dir, datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))

        if not FLAGS.train_db and not FLAGS.validation_db and not FLAGS.inference_db and not FLAGS.visualizeModelPath:
            logging.error("At least one of the following file sources should be specified: "
                          "train_db, validation_db or inference_db")
            exit(-1)

        if FLAGS.seed:
            tf.set_random_seed(FLAGS.seed)

        batch_size_train = FLAGS.batch_size
        batch_size_val = FLAGS.batch_size
        logging.info("Train batch size is %s and validation batch size is %s", batch_size_train, batch_size_val)

        # This variable keeps track of next epoch, when to perform validation.
        next_validation = FLAGS.validation_interval
        logging.info("Training epochs to be completed for each validation : %s", next_validation)

        # This variable keeps track of next epoch, when to save model weights.
        next_snapshot_save = FLAGS.snapshotInterval
        logging.info("Training epochs to be completed before taking a snapshot : %s", next_snapshot_save)
        last_snapshot_save_epoch = 0

        snapshot_prefix = FLAGS.snapshotPrefix if FLAGS.snapshotPrefix else FLAGS.network.split('.')[0]
        logging.info("Model weights will be saved as %s_<EPOCH>_Model.ckpt", snapshot_prefix)

        if not os.path.exists(FLAGS.save):
            os.makedirs(FLAGS.save)
            logging.info("Created a directory %s to save all the snapshots", FLAGS.save)

        # Load mean variable
        if FLAGS.subtractMean == 'none':
            mean_loader = None
        else:
            if not FLAGS.mean:
                logging.error("subtractMean parameter not set to 'none' yet mean image path is unset")
                exit(-1)
            logging.info("Loading mean tensor from %s file", FLAGS.mean)
            mean_loader = tf_data.MeanLoader(FLAGS.mean, FLAGS.subtractMean, FLAGS.bitdepth)

        classes = 0
        nclasses = 0
        if FLAGS.labels_list:
            logging.info("Loading label definitions from %s file", FLAGS.labels_list)
            classes = loadLabels(FLAGS.labels_list)
            nclasses = len(classes)
            if not classes:
                logging.error("Reading labels file %s failed.", FLAGS.labels_list)
                exit(-1)
            logging.info("Found %s classes", nclasses)

        # Create a data-augmentation dict
        aug_dict = {
            'aug_flip': FLAGS.augFlip,
            'aug_noise': FLAGS.augNoise,
            'aug_contrast': FLAGS.augContrast,
            'aug_whitening': FLAGS.augWhitening,
            'aug_HSV': {
                'h': FLAGS.augHSVh,
                's': FLAGS.augHSVs,
                'v': FLAGS.augHSVv,
            },
        }

        # Import the network file
        path_network = os.path.join(os.path.dirname(os.path.realpath(__file__)), FLAGS.networkDirectory, FLAGS.network)
        exec(open(path_network).read(), globals())

        try:
            UserModel
        except NameError:
            logging.error("The user model class 'UserModel' is not defined.")
            exit(-1)
        if not inspect.isclass(UserModel):  # noqa
            logging.error("The user model class 'UserModel' is not a class.")
            exit(-1)
        # @TODO(tzaman) - add mode checks to UserModel

        if FLAGS.train_db:
            with tf.name_scope(digits.STAGE_TRAIN) as stage_scope:
                train_model = Model(digits.STAGE_TRAIN, FLAGS.croplen, nclasses, FLAGS.optimization, FLAGS.momentum)
                train_model.create_dataloader(FLAGS.train_db)
                train_model.dataloader.setup(FLAGS.train_labels,
                                             FLAGS.shuffle,
                                             FLAGS.bitdepth,
                                             batch_size_train,
                                             FLAGS.epoch,
                                             FLAGS.seed)
                train_model.dataloader.set_augmentation(mean_loader, aug_dict)
                train_model.create_model(UserModel, stage_scope)  # noqa

        if FLAGS.validation_db:
            with tf.name_scope(digits.STAGE_VAL) as stage_scope:
                val_model = Model(digits.STAGE_VAL, FLAGS.croplen, nclasses, reuse_variable=True)
                val_model.create_dataloader(FLAGS.validation_db)
                val_model.dataloader.setup(FLAGS.validation_labels,
                                           False,
                                           FLAGS.bitdepth,
                                           batch_size_val,
                                           1e9,
                                           FLAGS.seed)  # @TODO(tzaman): set numepochs to 1
                val_model.dataloader.set_augmentation(mean_loader)
                val_model.create_model(UserModel, stage_scope)  # noqa

        if FLAGS.inference_db:
            with tf.name_scope(digits.STAGE_INF) as stage_scope:
                inf_model = Model(digits.STAGE_INF, FLAGS.croplen, nclasses)
                inf_model.create_dataloader(FLAGS.inference_db)
                inf_model.dataloader.setup(None, False, FLAGS.bitdepth, FLAGS.batch_size, 1, FLAGS.seed)
                inf_model.dataloader.set_augmentation(mean_loader)
                inf_model.create_model(UserModel, stage_scope)  # noqa

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
                          allow_soft_placement=True,  # will automatically do non-gpu supported ops on cpu
                          inter_op_parallelism_threads=TF_INTER_OP_THREADS,
                          intra_op_parallelism_threads=TF_INTRA_OP_THREADS,
                          log_device_placement=FLAGS.log_device_placement))

        if FLAGS.visualizeModelPath:
            visualize_graph(sess.graph_def, FLAGS.visualizeModelPath)
            exit(0)

        # Saver creation.
        if FLAGS.save_vars == 'all':
            vars_to_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        elif FLAGS.save_vars == 'trainable':
            vars_to_save = tf.all_variables()
        else:
            logging.error('Unknown save_var flag (%s)' % FLAGS.save_vars)
            exit(-1)
        saver = tf.train.Saver(vars_to_save, max_to_keep=0, sharded=FLAGS.serving_export)

        # Initialize variables
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        # If weights option is set, preload weights from existing models appropriately
        if FLAGS.weights:
            load_snapshot(sess, FLAGS.weights, tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        # Tensorboard: Merge all the summaries and write them out
        writer = tf.summary.FileWriter(os.path.join(FLAGS.summaries_dir, 'tb'), sess.graph)

        # If we are inferencing, only do that.
        if FLAGS.inference_db:
            inf_model.start_queue_runners(sess)
            Inference(sess, inf_model)

        queue_size_op = []
        for n in tf.get_default_graph().as_graph_def().node:
            if '_Size' in n.name:
                queue_size_op.append(n.name+':0')

        start = time.time()  # @TODO(tzaman) - removeme

        # Initial Forward Validation Pass
        if FLAGS.validation_db:
            val_model.start_queue_runners(sess)
            Validation(sess, val_model, 0)

        if FLAGS.train_db:
            # During training, a log output should occur at least X times per epoch or every X images, whichever lower
            train_steps_per_epoch = train_model.dataloader.get_total() / batch_size_train
            if math.ceil(train_steps_per_epoch/MIN_LOGS_PER_TRAIN_EPOCH) < math.ceil(5000/batch_size_train):
                logging_interval_step = int(math.ceil(train_steps_per_epoch/MIN_LOGS_PER_TRAIN_EPOCH))
            else:
                logging_interval_step = int(math.ceil(5000/batch_size_train))
            logging.info("During training. details will be logged after every %s steps (batches)",
                         logging_interval_step)

            # epoch value will be calculated for every batch size. To maintain unique epoch value between batches,
            # it needs to be rounded to the required number of significant digits.
            epoch_round = 0  # holds the required number of significant digits for round function.
            tmp_batchsize = batch_size_train*logging_interval_step
            while tmp_batchsize <= train_model.dataloader.get_total():
                tmp_batchsize = tmp_batchsize * 10
                epoch_round += 1
            logging.info("While logging, epoch value will be rounded to %s significant digits", epoch_round)

            # Create the learning rate policy
            total_training_steps = train_model.dataloader.num_epochs * train_model.dataloader.get_total() / \
                train_model.dataloader.batch_size
            lrpolicy = lr_policy.LRPolicy(FLAGS.lr_policy,
                                          FLAGS.lr_base_rate,
                                          FLAGS.lr_gamma,
                                          FLAGS.lr_power,
                                          total_training_steps,
                                          FLAGS.lr_stepvalues)
            train_model.start_queue_runners(sess)

            # Training
            logging.info('Started training the model')

            current_epoch = 0
            try:
                step = 0
                step_last_log = 0
                print_vals_sum = 0
                while not train_model.queue_coord.should_stop():
                    log_runtime = FLAGS.log_runtime_stats_per_step and (step % FLAGS.log_runtime_stats_per_step == 0)

                    run_options = None
                    run_metadata = None
                    if log_runtime:
                        # For a HARDWARE_TRACE you need NVIDIA CUPTI, a 'CUDA-EXTRA'
                        # SOFTWARE_TRACE HARDWARE_TRACE FULL_TRACE
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()

                    feed_dict = {train_model.learning_rate: lrpolicy.get_learning_rate(step)}

                    if False:
                        for op in train_model.train:
                            _, summary_str, step = sess.run([op, train_model.summary, train_model.global_step],
                                                            feed_dict=feed_dict,
                                                            options=run_options,
                                                            run_metadata=run_metadata)
                    else:
                        _, summary_str, step = sess.run([train_model.train,
                                                         train_model.summary,
                                                         train_model.global_step],
                                                        feed_dict=feed_dict,
                                                        options=run_options,
                                                        run_metadata=run_metadata)

                    # HACK
                    step = step / len(train_model.train)

                    # logging.info(sess.run(queue_size_op)) # DEVELOPMENT: for checking the queue size

                    if log_runtime:
                        writer.add_run_metadata(run_metadata, str(step))
                        save_timeline_trace(run_metadata, FLAGS.save, int(step))

                    writer.add_summary(summary_str, step)

                    # Parse the summary
                    tags, print_vals = summary_to_lists(summary_str)

                    print_vals_sum = print_vals + print_vals_sum

                    # @TODO(tzaman): account for variable batch_size value on very last epoch
                    current_epoch = round((step * batch_size_train) / train_model.dataloader.get_total(), epoch_round)
                    # Start with a forward pass
                    if ((step % logging_interval_step) == 0):
                        steps_since_log = step - step_last_log
                        print_list = print_summarylist(tags, print_vals_sum/steps_since_log)
                        logging.info("Training (epoch " + str(current_epoch) + "): " + print_list)
                        print_vals_sum = 0
                        step_last_log = step

                    # Potential Validation Pass
                    if FLAGS.validation_db and current_epoch >= next_validation:
                        Validation(sess, val_model, current_epoch)
                        # Find next nearest epoch value that exactly divisible by FLAGS.validation_interval:
                        next_validation = (round(float(current_epoch)/FLAGS.validation_interval) + 1) * \
                            FLAGS.validation_interval

                    # Saving Snapshot
                    if FLAGS.snapshotInterval > 0 and current_epoch >= next_snapshot_save:
                        checkpoint_path, graphdef_path = save_snapshot(sess,
                                                                       saver,
                                                                       FLAGS.save,
                                                                       snapshot_prefix,
                                                                       current_epoch,
                                                                       FLAGS.serving_export
                                                                       )

                        # To find next nearest epoch value that exactly divisible by FLAGS.snapshotInterval
                        next_snapshot_save = (round(float(current_epoch)/FLAGS.snapshotInterval) + 1) * \
                            FLAGS.snapshotInterval
                        last_snapshot_save_epoch = current_epoch
                    writer.flush()

            except tf.errors.OutOfRangeError:
                logging.info('Done training for epochs: tf.errors.OutOfRangeError')
            except ValueError as err:
                logging.error(err.args[0])
                exit(-1)  # DIGITS wants a dirty error.
            except (KeyboardInterrupt):
                logging.info('Interrupt signal received.')

            # If required, perform final snapshot save
            if FLAGS.snapshotInterval > 0 and FLAGS.epoch > last_snapshot_save_epoch:
                checkpoint_path, graphdef_path =\
                    save_snapshot(sess, saver, FLAGS.save, snapshot_prefix, FLAGS.epoch, FLAGS.serving_export)

        print('Training wall-time:', time.time()-start)  # @TODO(tzaman) - removeme

        # If required, perform final Validation pass
        if FLAGS.validation_db and current_epoch >= next_validation:
            Validation(sess, val_model, current_epoch)

        if FLAGS.train_db:
            if FLAGS.labels_list:
                output_tensor = train_model.towers[0].inference
                out_name, _ = output_tensor.name.split(':')

        if FLAGS.train_db:
            del train_model
        if FLAGS.validation_db:
            del val_model
        if FLAGS.inference_db:
            del inf_model

        # We need to call sess.close() because we've used a with block
        sess.close()

        writer.close()

    tf.reset_default_graph()

    del sess
    if FLAGS.train_db:
        if FLAGS.labels_list:
            path_frozen = os.path.join(FLAGS.save, 'frozen_model.pb')
            print('Saving frozen model at path {}'.format(path_frozen))
            freeze_graph.freeze_graph(
                input_graph=graphdef_path,
                input_saver='',
                input_binary=True,
                input_checkpoint=checkpoint_path,
                output_node_names=out_name,
                restore_op_name="save/restore_all",
                filename_tensor_name="save/Const:0",
                output_graph=path_frozen,
                clear_devices=True,
                initializer_nodes="",
            )

    logging.info('END')

    exit(0)

if __name__ == '__main__':
    tf.app.run()
