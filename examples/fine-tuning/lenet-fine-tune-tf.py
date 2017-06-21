class UserModel(Tower):

    @model_property
    def inference(self):

        # Create some wrappers for simplicity
        def conv2d(x, W, b, s, name, padding='SAME'):
            # Conv2D wrapper, with bias and relu activation
            x = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding=padding, name=name + '_conv2d')
            x = tf.nn.bias_add(x, b, name=name + 'bias_add')
            return tf.nn.relu(x)

        def maxpool2d(x, k, s, name, padding='VALID'):
            # MaxPool2D wrapper
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding=padding, name=name + '_maxpool2d')

        # Create model
        def conv_net(x, weights, biases):
            # scale (divide by MNIST std)
            x = x * 0.0125

            # Convolution Layer
            conv1 = conv2d(x, weights['wc1'], biases['bc1'], s=1, name='CONV1', padding='VALID')
            # Max Pooling (down-sampling)
            conv1 = maxpool2d(conv1, k=2, s=2, name='CONV1', padding='VALID')

            # Convolution Layer
            conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], s=1, name='CONV2', padding='VALID')
            # Max Pooling (down-sampling)
            conv2 = maxpool2d(conv2, k=2, s=2, name='CONV2', padding='VALID')

            # Fully connected layer
            # Reshape conv2 output to fit fully connected layer input
            fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]], name="FC1_reshape")
            fc1 = tf.add(tf.matmul(fc1, weights['wd1'], name='FC1_multi'), biases['bd1'], name='FC1_add')
            fc1 = tf.nn.relu(fc1, name='FC1_relu')

            # Apply Dropout
            if self.is_training:
                fc1 = tf.nn.dropout(fc1, 0.5, name='FC1_drop')

            # Output, class prediction
            out = tf.add(tf.matmul(fc1, weights['out'], name='OUT_multi'), biases['out'], name='OUT_add')
            
            true_out = tf.add(tf.matmul(out, weights['true_out'], name='OUT_multi'), biases['true_out'], name='TRUE_OUT_add')
            return true_out

        # Store layers weight & bias
        weights = {
            # 5x5 conv, 1 input, 20 outputs
            'wc1': tf.get_variable('wc1', [5, 5, self.input_shape[2], 20], initializer=tf.contrib.layers.xavier_initializer()),
            # 5x5 conv, 20 inputs, 50 outputs
            'wc2': tf.get_variable('wc2', [5, 5, 20, 50], initializer=tf.contrib.layers.xavier_initializer()),
            # fully connected, 4*4*16=800 inputs, 500 outputs
            'wd1': tf.get_variable('wd1', [4*4*50, 500], initializer=tf.contrib.layers.xavier_initializer()),
            # 500 inputs, 10 outputs (class prediction)
            'out': tf.get_variable('wout_not_in_use', [500, 10], initializer=tf.contrib.layers.xavier_initializer()),
            # adjust from 10 classes to 2 output
            'true_out': tf.get_variable('twout', [10, 2], initializer=tf.contrib.layers.xavier_initializer())
        }

        self.weights = weights

        # Leave the intial biases zero
        biases = {
            'bc1': tf.get_variable('bc1', [20], initializer=tf.constant_initializer(0.0)),
            'bc2': tf.get_variable('bc2', [50], initializer=tf.constant_initializer(0.0)),
            'bd1': tf.get_variable('bd1', [500], initializer=tf.constant_initializer(0.0)),
            'out': tf.get_variable('bout_not_in_use', [10], initializer=tf.constant_initializer(0.0)),
            'true_out': tf.get_variable('tbout', [2], initializer=tf.constant_initializer(0.0))
        }

        self.biases = biases

        model = conv_net(self.x, weights, biases)
        return model

    @model_property
    def loss(self):
        loss = digits.classification_loss(self.inference, self.y)
        accuracy = digits.classification_accuracy(self.inference, self.y)
        self.summaries.append(tf.summary.scalar(accuracy.op.name, accuracy))
        return loss
        