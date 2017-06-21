class UserModel(Tower):

    @model_property
    def inference(self):

        assert self.input_shape[0]==224, 'Input shape should be 224 pixels'
        assert self.input_shape[1]==224, 'Input shape should be 224 pixels'

        # Create some wrappers for simplicity
        def conv2d(x, W, b, s, padding='SAME'):
            # Conv2D wrapper, with bias and relu activation
            x = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding=padding)
            x = tf.nn.bias_add(x, b)
            return tf.nn.relu(x)

        def maxpool2d(x, k, s, padding='VALID'):
            # MaxPool2D wrapper
            return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, s, s, 1], padding=padding)

        # Create model
        def conv_net(x, weights, biases):
            conv1 = conv2d(x, weights['wc1'], biases['bc1'], s=4, padding='SAME')
            #conv1 = tf.nn.local_response_normalization(conv1, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0)
            pool1 = maxpool2d(conv1, k=3, s=2)
            conv2 = conv2d(pool1, weights['wc2'], biases['bc2'], s=1, padding='SAME')
            #conv2 = tf.nn.local_response_normalization(conv2, depth_radius=2, alpha=2e-5, beta=0.75, bias=1.0)
            pool2 = maxpool2d(conv2, k=3, s=2)
            conv3 = conv2d(pool2, weights['wc3'], biases['bc3'], s=1, padding='SAME')
            conv4 = conv2d(conv3, weights['wc4'], biases['bc4'], s=1, padding='SAME')
            conv5 = conv2d(conv4, weights['wc5'], biases['bc5'], s=1, padding='SAME')
            pool5 = maxpool2d(conv5, k=3, s=2)

            # Flatten
            flatten = tf.reshape(pool5, [-1, weights['wd1'].get_shape().as_list()[0]])
            
            fc1 = tf.add(tf.matmul(flatten, weights['wd1']), biases['bd1'])
            fc1 = tf.nn.relu(fc1)
            if self.is_training:
                fc1 = tf.nn.dropout(fc1, 0.5)

            fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
            fc2 = tf.nn.relu(fc2)
            if self.is_training:
                fc2 = tf.nn.dropout(fc2, 0.5)
            
            # Output, class prediction
            out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])
            return out


        # Initialize W using stddev 1/sqrt(n), with n the input dimension size.
        # Store layers weight & bias
        weights = {
            # 11x11 conv, #channels input, 96 outputs
            'wc1': tf.get_variable('wc1', [11, 11, self.input_shape[2], 96], initializer=tf.contrib.layers.xavier_initializer()),
            # 5x5 conv, 96 inputs, 256 outputs
            'wc2': tf.get_variable('wc2', [5, 5, 96, 256], initializer=tf.contrib.layers.xavier_initializer()),
            # 3x3 conv, 256 inputs, 384 outputs
            'wc3': tf.get_variable('wc3', [3, 3, 256, 384], initializer=tf.contrib.layers.xavier_initializer()),
            # 3x3 conv, 384 inputs, 384 outputs
            'wc4': tf.get_variable('wc4', [3, 3, 384, 384], initializer=tf.contrib.layers.xavier_initializer()),
            # 3x3 conv, 384 inputs, 256 outputs
            'wc5': tf.get_variable('wc5', [3, 3, 384, 256], initializer=tf.contrib.layers.xavier_initializer()),

            # fully connected, 6*6*256=9216 inputs, 4096 outputs
            'wd1': tf.get_variable('wd1', [6*6*256, 4096], initializer=tf.contrib.layers.xavier_initializer()),
            # fully connected, 4096 in, 4096 out
            'wd2': tf.get_variable('wd2', [4096, 4096], initializer=tf.contrib.layers.xavier_initializer()),
            # 4096 inputs, #classes outputs (class prediction)
            'out': tf.get_variable('wout', [4096, self.nclasses], initializer=tf.contrib.layers.xavier_initializer()),
        }

        # Leave the intial biases zero
        biases = {
            'bc1': tf.get_variable('bc1', [96], initializer=tf.constant_initializer(0.0)),
            'bc2': tf.get_variable('bc2', [256], initializer=tf.constant_initializer(0.0)),
            'bc3': tf.get_variable('bc3', [384], initializer=tf.constant_initializer(0.0)),
            'bc4': tf.get_variable('bc4', [384], initializer=tf.constant_initializer(0.0)),
            'bc5': tf.get_variable('bc5', [256], initializer=tf.constant_initializer(0.0)),
            'bd1': tf.get_variable('bd1', [4096], initializer=tf.constant_initializer(0.0)),
            'bd2': tf.get_variable('bd2', [4096], initializer=tf.constant_initializer(0.0)),
            'out': tf.get_variable('bout', [self.nclasses], initializer=tf.constant_initializer(0.0))
        }

        model = conv_net(self.x, weights, biases)
        return model

    @model_property
    def loss(self):
        loss = digits.classification_loss(self.inference, self.y)
        accuracy = digits.classification_accuracy(self.inference, self.y)
        self.summaries.append(tf.summary.scalar(accuracy.op.name, accuracy))
        return loss
