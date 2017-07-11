# Tensorflow MNIST autoencoder model using TensorFlow-Slim
from model import Tower
from utils import model_property
import tensorflow as tf
import tensorflow.contrib.slim as slim
import utils as digits


class UserModel(Tower):

    @model_property
    def inference(self):

        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005)):
            const = tf.constant(0.00390625)
            model = tf.multiply(self.x, const)
            model = tf.reshape(model, shape=[-1, 784])  # equivalent to `model = slim.flatten(_x)`
            model = slim.fully_connected(model, 300, scope='fc1')
            model = slim.fully_connected(model, 50, scope='fc2')
            model = slim.fully_connected(model, 300, scope='fc3')
            model = slim.fully_connected(model, 784, activation_fn=None, scope='fc4')
            model = tf.reshape(model, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        # The below image summary makes it very easy to review your result
        tf.summary.image(self.x.op.name, self.x, max_outputs=5, collections=['summaries'])
        tf.summary.image(model.op.name, model, max_outputs=5, collections=['summaries'])

        return model

    @model_property
    def loss(self):
        return digits.mse_loss(self.inference, self.x)
