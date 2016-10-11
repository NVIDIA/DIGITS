def build_model(params):
    _x = tf.reshape(params['x'], shape=[-1, params['input_shape'][0], params['input_shape'][1], params['input_shape'][2]])
    #tf.image_summary(_x.op.name, _x, max_images=10, collections=[digits.GraphKeys.SUMMARIES_TRAIN])

    # Split out the channel in two
    lhs, rhs = tf.split(0, 2, _x, name='split_batch')

    with slim.arg_scope([slim.conv2d, slim.fully_connected], 
            weights_initializer=tf.contrib.layers.xavier_initializer(),
            weights_regularizer=slim.l2_regularizer(0.0005) ):
        with tf.variable_scope("siamese") as scope:
            def make_tower(net):
                net = slim.conv2d(net, 20, [5, 5], padding='VALID', scope='conv1')
                net = slim.max_pool2d(net, [2, 2], padding='VALID', scope='pool1')
                net = slim.conv2d(net, 50, [5, 5], padding='VALID', scope='conv2')
                net = slim.max_pool2d(net, [2, 2], padding='VALID', scope='pool2')
                net = slim.flatten(net)
                net = slim.fully_connected(net, 500, scope='fc1')
                net = slim.fully_connected(net, 2, activation_fn=None, scope='fc2')
                return net

            lhs = make_tower(lhs)
            lhs = tf.reshape(lhs, shape=[-1, 2])
            scope.reuse_variables()
            rhs = make_tower(rhs)
            rhs = tf.reshape(rhs, shape=[-1, 2])

    def loss(y):
        y = tf.reshape(y, shape=[-1])
        ylhs, yrhs = tf.split(0, 2, y, name='split_label')
        y = tf.equal(ylhs, yrhs)
        y = tf.to_float(y)
        return digits.constrastive_loss(lhs, rhs, y)

    return {
        'model' : tf.concat(0, [lhs, rhs]),
        'loss' : loss,
        }
