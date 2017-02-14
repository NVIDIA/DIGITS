import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter


class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon  = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    batch_size = tf.shape(x)[0]
    return tf.concat(3, [x, y*tf.ones([batch_size, int(x_shapes[1]), int(x_shapes[2]), int(y_shapes[3])])])

def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.bias_add(conv, biases)

        return conv


def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), output_shape)

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

""" Main Model Class """
class UserModel(Tower):

    def __init__(self, *args, **kwargs):
        """Identify the correct input nodes."""
        super(UserModel, self).__init__(*args, **kwargs)

        self.dcgan_init(image_size=28,
                        y_dim=10,
                        output_size=28,
                        c_dim=1,
                        is_crop=False,
                        )

    @model_property
    def inference(self):
        """ op to use for inference """
        images = self.G * 255
        images_flat = tf.reshape(images, [self.batch_size, self.image_size * self.image_size * self.c_dim])
        zgen_flat = tf.reshape(self.DzGEN, [self.batch_size, self.z_dim])
        return tf.concat(1, [zgen_flat, images_flat])

    @model_property
    def loss(self):
        """ return list of losses (and corresponding vars) to optimize """
        losses = [
            {'loss': self.dzgen_loss, 'vars': self.d_vars},
        ]
        return losses

    def dcgan_init(self, image_size=108, is_crop=True,
                   output_size=64, y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                   gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                   ):
        """

        Args:
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.is_crop = is_crop
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.output_size = output_size

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim

        self.batch_size = tf.shape(self.x)[0]

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.build_model()

    def build_model(self):

        if self.is_inference:
            self.y = tf.to_int32(3*tf.ones(shape=[self.batch_size]))

        x_reshaped = tf.reshape(self.x, shape=[self.batch_size, self.image_size, self.image_size, self.c_dim], name='x_reshaped')
        self.images = x_reshaped / 255.

        if self.y_dim:
            self.y = tf.one_hot(self.y, self.y_dim, name='y_onehot')
            self.DzGEN, self.D_logits  = self.discriminator(self.images, self.y, reuse=False)
            self.G = self.generator(self.DzGEN, self.y)
        else:
            raise NotImplementedError()

        self.summaries.append(image_summary("G", self.G, max_outputs=5))
        self.summaries.append(image_summary("X", self.images, max_outputs=5))
        self.summaries.append(histogram_summary("G_hist", self.G))
        self.summaries.append(histogram_summary("X_hist", self.images))

        self.dzgen_loss = tf.reduce_mean(tf.square(self.G - self.images), name="loss_DzGEN")

        self.summaries.append(scalar_summary("DzGen_loss", self.dzgen_loss))

        t_vars = tf.trainable_variables()

        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.d_vars = [var for var in t_vars if 'd_' in var.name]

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            if not self.y_dim:
                raise NotImplementedError()
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv'), train=self.is_training))
                sz = h1.get_shape()
                h1 = tf.reshape(h1, [self.batch_size, int(sz[1] * sz[2] * sz[3])])
                h1 = tf.concat(1, [h1, y])

                h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin'), train=self.is_training))
                h2 = tf.concat(1, [h2, y])

                h3 = linear(h2, self.z_dim, 'd_h3_lin_retrain')
                return h3, h3


    def generator(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            if not self.y_dim:
                raise NotImplementedError()
            else:
                s = self.output_size
                s2, s4 = int(s/2), int(s/4)

                # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = tf.concat(1, [z, y])

                h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
                h0 = tf.concat(1, [h0, y])

                h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin'), train=False))
                h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])

                h1 = conv_cond_concat(h1, yb)

                h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2'), train=False))
                h2 = conv_cond_concat(h2, yb)

                return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))
