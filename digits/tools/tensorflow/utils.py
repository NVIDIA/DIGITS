# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
#
# This document should comply with PEP-8 Style Guide
# Linter: pylint

"""
Digits default Tensorflow Ops as helper functions.

"""

import functools
import tensorflow as tf
from tensorflow.python.client import device_lib

STAGE_TRAIN = 'train'
STAGE_VAL = 'val'
STAGE_INF = 'inf'


class GraphKeys(object):
    TEMPLATE = "model"
    QUEUE_RUNNERS = "queue_runner"
    MODEL = "model"
    LOSS = "loss"  # The namescope
    LOSSES = "losses"  # The collection
    LOADER = "data"


def model_property(function):
    # From https://danijar.com/structuring-your-tensorflow-models/
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


def classification_loss(pred, y):
    """
    Definition of the loss for regular classification
    """
    ssoftmax = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y, name='cross_entropy_single')
    return tf.reduce_mean(ssoftmax, name='cross_entropy_batch')


def mse_loss(lhs, rhs):
    return tf.reduce_mean(tf.square(lhs - rhs))


def constrastive_loss(lhs, rhs, y, margin=1.0):
    """
    Contrastive loss confirming to the Caffe definition
    """
    d = tf.reduce_sum(tf.square(tf.subtract(lhs, rhs)), 1)
    d_sqrt = tf.sqrt(1e-6 + d)
    loss = (y * d) + ((1 - y) * tf.square(tf.maximum(margin - d_sqrt, 0)))
    return tf.reduce_mean(loss)  # Note: constant component removed (/2)


def classification_accuracy_top_n(pred, y, top_n):
    single_acc_t = tf.nn.in_top_k(pred, y, top_n)
    return tf.reduce_mean(tf.cast(single_acc_t, tf.float32), name='accuracy_top_%d' % top_n)


def classification_accuracy(pred, y):
    """
    Default definition of accuracy. Something is considered accurate if and only
    if a true label exactly matches the highest value in the prediction vector.
    """
    single_acc = tf.equal(y, tf.argmax(pred, 1))
    return tf.reduce_mean(tf.cast(single_acc, tf.float32), name='accuracy')


def nhwc_to_nchw(x):
    return tf.transpose(x, [0, 3, 1, 2])


def hwc_to_chw(x):
    return tf.transpose(x, [2, 0, 1])


def nchw_to_nhwc(x):
    return tf.transpose(x, [0, 2, 3, 1])


def chw_to_hwc(x):
    return tf.transpose(x, [1, 2, 0])


def bgr_to_rgb(x):
    return tf.reverse(x, [2])


def rgb_to_bgr(x):
    return tf.reverse(x, [2])


def get_available_gpus():
    """
    Queries the CUDA GPU devices visible to Tensorflow.
    Returns:
        A list with tf-style gpu strings (f.e. ['/gpu:0', '/gpu:1'])
    """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
