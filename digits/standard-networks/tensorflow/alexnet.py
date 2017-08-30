# Preferred settings for this model is:
# Base Learning Rate = 0.001
# Crop Size = 224

from model import Tower
from utils import model_property
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils as digits


class UserModel(Tower):

    @model_property
    def inference(self):
        x = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(1e-6)):
            model = slim.conv2d(x, 96, [11, 11], 4, padding='VALID', scope='conv1')
            model = slim.max_pool2d(model, [3, 3], 2, scope='pool1')
            model = slim.conv2d(model, 256, [5, 5], 1, scope='conv2')
            model = slim.max_pool2d(model, [3, 3], 2, scope='pool2')
            model = slim.conv2d(model, 384, [3, 3], 1, scope='conv3')
            model = slim.conv2d(model, 384, [3, 3], 1, scope='conv4')
            model = slim.conv2d(model, 256, [3, 3], 1, scope='conv5')
            model = slim.max_pool2d(model, [3, 3], 2, scope='pool5')
            model = slim.flatten(model)
            model = slim.fully_connected(model, 4096, activation_fn=None, scope='fc1')
            model = slim.dropout(model, 0.5, is_training=self.is_training, scope='do1')
            model = slim.fully_connected(model, 4096, activation_fn=None, scope='fc2')
            model = slim.dropout(model, 0.5, is_training=self.is_training, scope='do2')
            model = slim.fully_connected(model, self.nclasses, activation_fn=None, scope='fc3')
        return model

    @model_property
    def loss(self):
        model = self.inference
        loss = digits.classification_loss(model, self.y)
        accuracy = digits.classification_accuracy(model, self.y)
        self.summaries.append(tf.summary.scalar(accuracy.op.name, accuracy))
        return loss
