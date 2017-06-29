# The MIT License (MIT)
#
# Original work Copyright (c) 2016 Taehoon Kim
# Modified work Copyright (c) 2017, NVIDIA CORPORATION.  All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf

image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter


class batch_norm(object):
    """
    This class creates an op that composes the specified tensor with a batch
    normalization layer.
    """

    def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
        """Instance initialization"""
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        """
        Functional interface

        Args:
            x: tensor to compose
            train: set to True during training and False otherwise
        """
        return tf.contrib.layers.batch_norm(x,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name)


def conv_cond_concat(x, y):
    """
    Concatenate conditioning matrix across channel axis.

    The specified input tensor is concatenated with K feature maps (K = number of classes)
    across the channel dimension. Each of the K feature maps is set to all-zeros except for
    the one whose index matches the target class (which is set to all-ones).

    Args:
        x: non-conditioned tensor. Shape: [N, H, W, C]
        y: one-hot encoded conditioning matrix. Shape: [N, K]

    Returns:
        conditioned feature map. Shape: [N, H, W, C + K]
    """
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    batch_size = tf.shape(x)[0]
    return tf.concat(3, [x, y * tf.ones([batch_size, int(x_shapes[1]), int(x_shapes[2]), int(y_shapes[3])])])


def conv2d(input_, output_dim,
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    """
    Compose specified symbol with 2D convolution layer

    Args:
        input_: tensor to compose. Shape: [N, H, W, C]
        output_dim: number of output features maps
        k_h: kernel height
        k_w: kernel width
        d_h: horizontal stride
        d_w: vertical stride
        stddev: standard deviation of gaussian distribution to use for random weight initialization
        name: name scope

    Returns:
        Composed tensor.
    """
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
    """
    Compose specified symbol with 2D *transpose* convolution layer

    Args:
        input_: tensor to compose. Shape: [N, H, W, C]
        output_shape: output shape
        k_h: kernel height
        k_w: kernel width
        d_h: horizontal stride
        d_w: vertical stride
        stddev: standard deviation of gaussian distribution to use for random weight initialization
        name: name scope

    Returns:
        Composed tensor.
    """
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w',
                            [k_h, k_w, output_shape[-1],
                             input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        deconv = tf.nn.conv2d_transpose(input_, w,
                                        output_shape=output_shape,
                                        strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), output_shape)

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def lrelu(x, leak=0.2, name="lrelu"):
    """Compose specified tensor with leaky Rectifier Linear Unit"""
    return tf.maximum(x, leak*x)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    """
    Compose specified tensor with linear (fully-connected) layer

    Args:
        input_: tensor to compose. Shape: [N, M]
        output_size: number of output neurons
        scope: name scope
        stddev: standard deviation of gaussian distribution to use for random weight initialization
        name: name scope
        with_w: whether to also return parameter variables

    Returns:
      Composed tensor. Shape: [N, output_size]
    """
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


class UserModel(Tower):
    """
    User Model definition

    DIGITS creates an instance of this class for every tower it needs
    to create. This includes:
    - one for training,
    - one for validation,
    - one for testing.

    In the case of multi-GPU training, one training instance is created
    for every GPU. DIGITS takes care of doing the gradient averaging
    across GPUs so this class only needs to define the inference op
    and desired loss/cost function.
    """

    def __init__(self, *args, **kwargs):
        """
        Identify the correct input nodes.

        In the parent class, DIGITS conveniently sets the following fields:
        - self.is_training: whether this is a training graph
        - self.is_inference: whether this graph is created for inference/testing
        - self.x: input node. Shape: [N, H, W, C]
        - self.y: label. Shape: [N] for scalar labels, [N, H, W, C] otherwise.
                  Only defined if self._is_training is True
        """
        super(UserModel, self).__init__(*args, **kwargs)

        # initialize graph with parameters for MNIST
        self.dcgan_init(image_size=28,
                        y_dim=10,
                        output_size=28,
                        c_dim=1)

    @model_property
    def inference(self):
        """op to use for inference"""

        # inference op is the output of the generator after rescaling
        # to the 8-bit range
        return tf.to_int32(self.G * 255)

    @model_property
    def loss(self):
        """
        Loss function

        Returns either an op or a list of dicts.
        If the returned value is an op then DIGITS will optimize against this op
        with respect to all trainable variables.
        If the returned value is a list then DIGITS will optimize against each
        loss in the list with respect to the specified variables.
        """

        # here we are returning a list because we want to alternately optimize the
        # discriminator on real samples, the discriminator on fake samples and the
        # generator.
        losses = [
            {'loss': self.d_loss_real, 'vars': self.d_vars},
            {'loss': self.d_loss_fake, 'vars': self.d_vars},
            {'loss': self.g_loss, 'vars': self.g_vars}
        ]
        return losses

    def dcgan_init(self, image_size=108,
                   output_size=64, y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                   gfc_dim=1024, dfc_dim=1024, c_dim=3):
        """
        Create the model

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

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        self.build_model()

    def build_model(self):
        """Create the main ops"""

        if not self.is_inference:
            # create both the generator and the discriminator
            # self.x is a batch of images - shape: [N, H, W, C]
            # self.y is a vector of labels - shape: [N]

            # sample z from a normal distribution
            self.z = tf.random_normal(shape=[self.batch_size, self.z_dim], dtype=tf.float32, seed=None, name='z')

            # rescale x to [0, 1]
            x_reshaped = tf.reshape(self.x, shape=[self.batch_size, self.image_size, self.image_size, self.c_dim],
                                    name='x_reshaped')
            self.images = x_reshaped / 255.

            # one hot encode the label - shape: [N] -> [N, self.y_dim]
            self.y = tf.one_hot(self.y, self.y_dim, name='y_onehot')

            # create the generator
            self.G = self.generator(self.z, self.y)

            # create one instance of the discriminator for real images (the input is
            # images from the dataset)
            self.D, self.D_logits = self.discriminator(self.images, self.y, reuse=False)

            # create another instance of the discriminator for fake images (the input is
            # the discriminator). Note how we are reusing variables to share weights between
            # both instances of the discriminator
            self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

            # aggregate losses across batch

            # we are using the cross entropy loss for all these losses
            d_real = tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D), name="loss_D_real")
            self.d_loss_real = tf.reduce_mean(d_real)
            d_fake = tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_), name="loss_D_fake")
            self.d_loss_fake = tf.reduce_mean(d_fake)
            self.d_loss = (self.d_loss_real + self.d_loss_fake) / 2.
            # the typical GAN set-up is that of a minimax game where D is trying to minimize
            # its own error and G is trying to maximize D's error however note how we are flipping G labels here:
            # instead of maximizing D's error, we are minimizing D's error on the 'wrong' label
            # this trick helps produce a stronger gradient
            g_loss = tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_), name="loss_G")
            self.g_loss = tf.reduce_mean(g_loss)

            # create some summaries for debug and monitoring
            self.summaries.append(histogram_summary("z", self.z))
            self.summaries.append(histogram_summary("d", self.D))
            self.summaries.append(histogram_summary("d_", self.D_))
            self.summaries.append(image_summary("G", self.G, max_outputs=5))
            self.summaries.append(image_summary("X", self.images, max_outputs=5))
            self.summaries.append(histogram_summary("G_hist", self.G))
            self.summaries.append(histogram_summary("X_hist", self.images))
            self.summaries.append(scalar_summary("d_loss_real", self.d_loss_real))
            self.summaries.append(scalar_summary("d_loss_fake", self.d_loss_fake))
            self.summaries.append(scalar_summary("g_loss", self.g_loss))
            self.summaries.append(scalar_summary("d_loss", self.d_loss))

            # all trainable variables
            t_vars = tf.trainable_variables()
            # G's variables
            self.g_vars = [var for var in t_vars if 'g_' in var.name]
            # D's variables
            self.d_vars = [var for var in t_vars if 'd_' in var.name]

            # Extra hook for debug: log chi-square distance between G's output histogram and the dataset's histogram
            value_range = [0.0, 1.0]
            nbins = 100
            hist_g = tf.histogram_fixed_width(self.G, value_range, nbins=nbins, dtype=tf.float32) / nbins
            hist_images = tf.histogram_fixed_width(self.images, value_range, nbins=nbins, dtype=tf.float32) / nbins
            chi_square = tf.reduce_mean(tf.div(tf.square(hist_g - hist_images), hist_g + hist_images))
            self.summaries.append(scalar_summary("chi_square", chi_square))
        else:
            # Create only the generator

            # self.x is the conditioned latent representation - shape: [self.batch_size, 1, self.z_dim + self.y_dim]
            self.x = tf.reshape(self.x, shape=[self.batch_size, self.z_dim + self.y_dim])
            # extract z and y
            self.y = self.x[:, self.z_dim:self.z_dim + self.y_dim]
            self.z = self.x[:, :self.z_dim]
            # create an instance of the generator
            self.G = self.generator(self.z, self.y)

    def discriminator(self, image, y=None, reuse=False):
        """
        Create the discriminator

        This creates a string of layers:
        - input - [N, 28, 28, 1]
        - concat conditioning - [N, 28, 28, 11]
        - conv layer with 11 5x5 kernels and 2x2 stride - [N, 14, 14, 11]
        - leaky relu - [N, 14, 14, 11]
        - concat conditioning - [N, 14, 14, 21]
        - conv layer with 74 5x5 kernels and 2x2 stride - [N, 7, 7, 74]
        - batch norm - [N, 14, 14, 64]
        - leaky relu - [N, 14, 14, 64]
        - flatten - [N, 3626]
        - concat conditioning - [N, 3636]
        - linear layer with 1014 output neurons - [N, 1024]
        - batch norm - [N, 1024]
        - leaky relu - [N, 1024]
        - concat conditioning - [N, 1034]
        - linear layer with 1 output neuron - [N, 1]

        Args:
            image: batch of input images - shape: [N, H, W, C]
            y: batch of one-hot encoded labels - shape: [N, K]
            reuse: whether to re-use previously created variables
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                # re-use (share) variables
                scope.reuse_variables()

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

            h3 = linear(h2, 1, 'd_h3_lin')

            return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        """
        Create the generator

        This creates a string of layers:
        - input - [N, 100]
        - concatenate conditioning - [N, 110]
        - linear layer with 1024 output neurons - [N, 1024]
        - batch norm - [N, 1024]
        - relu - [N, 1024]
        - concatenate conditioning - [N, 1034]
        - linear layer with 7*7*128=6272 output neurons - [N, 6272]
        - reshape 7x7 feature maps - [N, 7, 7, 128]
        - concatenate conditioning - [N, 7, 7, 138]
        - transpose convolution with 128 filters and stride 2 - [N, 14, 14, 128]
        - batch norm - [N, 14, 14, 128]
        - relu - [N, 14, 14, 128]
        - concatenate conditioing - [N, 14, 14, 138]
        - transpose convolution with 1 filter and stride 2 - [N, 28, 28, 1]
        """
        with tf.variable_scope("generator") as scope:

            s = self.output_size
            s2, s4 = int(s/2), int(s/4)

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = tf.concat(1, [z, y])

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=self.is_training))
            h0 = tf.concat(1, [h0, y])

            h1 = tf.nn.relu(self.g_bn1(linear(h0, self.gf_dim*2*s4*s4, 'g_h1_lin'), train=self.is_training))
            h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])

            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2'),
                                       train=self.is_training))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))
