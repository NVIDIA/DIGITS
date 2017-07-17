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
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            model = slim.repeat(x, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            model = slim.max_pool2d(model, [2, 2], scope='pool1')
            model = slim.repeat(model, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            model = slim.max_pool2d(model, [2, 2], scope='pool2')
            model = slim.repeat(model, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            model = slim.max_pool2d(model, [2, 2], scope='pool3')
            model = slim.repeat(model, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            model = slim.max_pool2d(model, [2, 2], scope='pool4')
            model = slim.repeat(model, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            model = slim.max_pool2d(model, [2, 2], scope='pool5')
            model = slim.flatten(model, scope='flatten5')
            model = slim.fully_connected(model, 4096, scope='fc6')
            model = slim.dropout(model, 0.5, is_training=self.is_training, scope='do6')
            model = slim.fully_connected(model, 4096, scope='fc7')
            model = slim.dropout(model, 0.5, is_training=self.is_training, scope='do7')
            model = slim.fully_connected(model, self.nclasses, activation_fn=None, scope='fcX8')
        return model

    @model_property
    def loss(self):
        loss = digits.classification_loss(self.inference, self.y)
        accuracy = digits.classification_accuracy(self.inference, self.y)
        self.summaries.append(tf.summary.scalar(accuracy.op.name, accuracy))
        return loss
