# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
#
# This document should comply with PEP-8 Style Guide
# Linter: pylint

"""
Interface for setting up a model in Tensorflow.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import logging
import tensorflow as tf

# Local imports
import tf_data
import utils as digits

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

# Constants
OUTPUT_HISTOGRAM_SUMMARIES = False # Very heavy for the CPU

def lazy_property(function):
    # From https://danijar.com/structuring-your-tensorflow-models/
    attribute = '_cache_' + function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

# -- from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py
def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(0, grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

class Model(object):
    """
    @TODO(tzaman)

    """
    def __init__(self, stage, croplen, nclasses):
        self.stage = stage
        self.croplen = croplen
        self.nclasses = nclasses
        self.dataloader = None
        self.optimization = None
        self.momentum = None

        self.queue_coord = None
        self.queue_threads = None

        self._summaries = []
        self.inference = None
        self.network_loss = None

        # Define graph keys in tf convention
        self.GraphKeys = {}
        self.GraphKeys['QUEUE_RUNNERS'] = "queue_runner_" + self.stage
        self.GraphKeys['MODEL'] = "model_" + self.stage
        self.GraphKeys['LOSS'] = "loss_" + self.stage # The name-scope
        self.GraphKeys['LOSSES'] = "losses" + self.stage # The collection
        self.GraphKeys['LOADER'] = "data_" + self.stage

        # Special exception for summaries, as they need to be accesible to the user model
        # in a tf compliant way
        if self.stage == digits.STAGE_TRAIN:
            self.GraphKeys['SUMMARIES'] = digits.GraphKeys.SUMMARIES_TRAIN
        elif self.stage == digits.STAGE_VAL:
            self.GraphKeys['SUMMARIES'] = digits.GraphKeys.SUMMARIES_VAL
        elif self.stage == digits.STAGE_INF:
            self.GraphKeys['SUMMARIES'] = digits.GraphKeys.SUMMARIES_INF

    def create_dataloader(self, db_path):
        self.dataloader = tf_data.LoaderFactory.set_source(db_path)
        self.dataloader.summaries = self._summaries
        self.dataloader.stage = self.stage
        self.dataloader.croplen = self.croplen
        self.dataloader.nclasses = self.nclasses

    def init_dataloader(self):
        with tf.device('/cpu:0'):
            with tf.name_scope(self.GraphKeys['LOADER']):
                self.dataloader.create_input_pipeline()

    def set_optimizer(self, optimization, momentum):
        self.optimization = optimization
        self.momentum = momentum
        # touch and initialize the optimizer and global_step
        self.global_step

    def create_model_from_template(self, network_template):

        available_devices = digits.get_available_gpus()
        if not available_devices:
            available_devices.append('/cpu:0')

        # Split the batch over the batch dimension over the number of available gpu's
        batch_x_split = tf.split(0, len(available_devices), self.dataloader.batch_x, name='split_batch')

        if self.stage != digits.STAGE_INF:
            # Inference never has labels
            batch_y_split = tf.split(0, len(available_devices), self.dataloader.batch_y, name='split_batch')

        # Run the user model through the build_model function that should be filled in
        grad_towers = []
        for gpu_id, gpu_device in enumerate(available_devices):
            with tf.device(gpu_device):
                with tf.name_scope('tower_%d' % gpu_id) as scope_tower:
                    with tf.name_scope(self.GraphKeys['MODEL']):
                        # Load the parameters to be  passed to the custom user network definition
                        model_params = {
                            'x' : batch_x_split[gpu_id],
                            'input_shape' : self.dataloader.get_shape(),
                            'nclasses' : self.nclasses,
                            'is_training' : self.stage == digits.STAGE_TRAIN,
                        }

                        user_network = network_template(model_params)

                        # Perform checks
                        if not user_network.has_key('model'):
                            logging.error("Model definition required in model file but not supplied.")
                            exit(-1)
                        else: # Key exists, check type
                            if 'tensorflow' not in str(type(user_network['model'])):
                                logging.error("Model definition required in model is not a tf operation type, but is type(%s)", type(user_network['model']))
                                exit(-1)

                        if not user_network.has_key('loss'):
                            logging.error("Loss function definition required in model file but not supplied.")
                            exit(-1)
                        else: # Key exists, check if callable
                            if not callable(user_network['loss']):
                                logging.error("Returned loss function should be a function, but is type(%s).", type(user_network['loss']))
                                exit(-1)

                        self.inference = user_network['model']

                    if self.stage == digits.STAGE_INF:
                        # For inferencing we will only use the inference part of the graph
                        continue;

                    with tf.name_scope(self.GraphKeys['LOSS']):

                        loss_op = user_network['loss'](batch_y_split[gpu_id])

                        tf.add_to_collection(self.GraphKeys['LOSSES'], loss_op)
                        #loss_op = tf.add_n(tf.get_collection(self.GraphKeys['LOSSES']), name='total_loss')
                        #tf.add_to_collection('losses', loss_op)

                        # Assemble all made within this scope so far (f.e. including potential L2-loss from user model)
                        total_tower_loss =tf.add_n(tf.get_collection(self.GraphKeys['LOSSES'], scope_tower), name='total_tower_loss')

                        if len(available_devices) > 1:
                            self._summaries.append(tf.scalar_summary('loss_t_%d' % gpu_id, total_tower_loss))


                    # Reuse the variables in this scope for the next tower/device
                    tf.get_variable_scope().reuse_variables()

                    if self.stage == digits.STAGE_TRAIN:
                        grad_tower = self.optimizer.compute_gradients(total_tower_loss)
                        grad_towers.append(grad_tower)

        if self.stage != digits.STAGE_INF:
            with tf.name_scope(self.GraphKeys['MODEL']):
                self._summaries.append(tf.scalar_summary('loss', tf.add_n(tf.get_collection(self.GraphKeys['LOSSES']))/len(available_devices)))

        # Assemble and average the gradients from all towers
        if self.stage == digits.STAGE_TRAIN:
            if len(grad_towers) == 1:
                grad_avg = grad_towers[0]
            else:
                grad_avg = average_gradients(grad_towers)
            apply_gradient_op = self.optimizer.apply_gradients(grad_avg, global_step=self.global_step)
            self.train = apply_gradient_op

    @lazy_property
    def summary(self):
        """
        Merge train summaries
        """

        # The below get_collection() commands retrieve any summaries that have been set by the user
        # in the model
        self._summaries += tf.get_collection(self.GraphKeys['SUMMARIES'],
                                             scope='.*'+self.GraphKeys['MODEL'])
        self._summaries += tf.get_collection(self.GraphKeys['SUMMARIES'],
                                             scope='.*'+self.GraphKeys['LOSS'])

        if not len(self._summaries):
            logging.error("No summaries defined. Please define at least one summary.")
            exit(-1)
        return tf.merge_summary(self._summaries)

    @lazy_property
    def global_step(self):
        # Force global_step on the CPU, becaues the GPU's first step will end at 0 instead of 1.
        with tf.device('/cpu:0'):
            return tf.get_variable('global_step', [], initializer=tf.constant_initializer(0),
                                   trainable=False)

    @lazy_property
    def learning_rate(self):
        # @TODO(tzaman): the learning rate is a function of the global step, so we could
        #  define it entirely in tf ops, instead of a placeholder and feeding.
        with tf.device('/cpu:0'):
            lr = tf.placeholder(tf.float32, shape=[], name='learning_rate')
            self._summaries.append(tf.scalar_summary('lr', lr))
            return lr

    @lazy_property
    def optimizer(self):
        logging.info("Optimizer:%s", self.optimization)
        if self.optimization == 'sgd':
            return tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        elif self.optimization == 'adadelta':
            return tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)
        elif self.optimization == 'adagrad':
            return tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        elif self.optimization == 'adagradda':
            return tf.train.AdagradDAOptimizer(learning_rate=self.learning_rate,
                                               global_step=self.global_step)
        elif self.optimization == 'momentum':
            return tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                              momentum=self.momentum)
        elif self.optimization == 'adam':
            return tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimization == 'ftrl':
            return tf.train.FtrlOptimizer(learning_rate=self.learning_rate)
        elif self.optimization == 'rmsprop':
            return tf.train.RMSPropOptimizer(learning_rate=self.learning_rate,
                                             momentum=self.momentum)
        else:
            logging.error("Invalid optimization flag %s", self.optimization)
            exit(-1)

    def start_queue_runners(self, sess):
        logging.info('Starting queue runners (%s)', self.stage)
        # Distinguish the queue runner collection (for easily obtaining them by collection key)
        queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
        for qr in queue_runners:
            if self.stage in qr.name:
                tf.add_to_collection(self.GraphKeys['QUEUE_RUNNERS'], qr)

        self.queue_coord = tf.train.Coordinator()
        self.queue_threads = tf.train.start_queue_runners(sess=sess, coord=self.queue_coord,
                                                          collection=self.GraphKeys['QUEUE_RUNNERS']
                                                         )
        logging.info('Queue runners started (%s)', self.stage)

    def __del__(self):
        # Destructor
        if self.queue_coord:
            # Close and terminate the queues
            self.queue_coord.request_stop()
            self.queue_coord.join(self.queue_threads)
