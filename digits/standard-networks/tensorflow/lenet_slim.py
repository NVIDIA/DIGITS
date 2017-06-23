class UserModel(Tower):

    @model_property
    def inference(self):
        x = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])
        # scale (divide by MNIST std)
        x = x * 0.0125
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005) ):
            model = slim.conv2d(x, 20, [5, 5], padding='VALID', scope='conv1')
            model = slim.max_pool2d(model, [2, 2], padding='VALID', scope='pool1')
            model = slim.conv2d(model, 50, [5, 5], padding='VALID', scope='conv2')
            model = slim.max_pool2d(model, [2, 2], padding='VALID', scope='pool2')
            model = slim.flatten(model)
            model = slim.fully_connected(model, 500, scope='fc1')
            model = slim.dropout(model, 0.5, is_training=self.is_training, scope='do1')
            model = slim.fully_connected(model, self.nclasses, activation_fn=None, scope='fc2')
            return model

    @model_property
    def loss(self):
        loss = digits.classification_loss(self.inference, self.y)
        accuracy = digits.classification_accuracy(self.inference, self.y)
        self.summaries.append(tf.scalar_summary(accuracy.op.name, accuracy))
        return loss
