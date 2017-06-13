# Tensorflow MNIST autoencoder model using TensorFlow-Slim

def build_model(params):

    _x = tf.reshape(params['x'], shape=[-1, params['input_shape'][0], params['input_shape'][1], params['input_shape'][2]])
    with slim.arg_scope([slim.fully_connected], 
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005) ):
        model = tf.reshape(_x, shape=[-1, 784]) # equivalent to `model = slim.flatten(_x)`
        model = slim.fully_connected(model, 300, scope='fc1')
        model = slim.fully_connected(model, 50, scope='fc2')
        model = slim.fully_connected(model, 300, scope='fc3')
        model = slim.fully_connected(model, 784, activation_fn=None, scope='fc4')
        model = tf.reshape(model, shape=[-1, params['input_shape'][0], params['input_shape'][1], params['input_shape'][2]])

    # The below image summary makes it very easy to review your result
    tf.image_summary(_x.op.name, _x, max_images=5, collections=[digits.GraphKeys.SUMMARIES_VAL])
    tf.image_summary(model.op.name, model, max_images=5, collections=[digits.GraphKeys.SUMMARIES_VAL])

    def loss(y):
        return digits.mse_loss(model, _x)

    return {
        'model' : model,
        'loss' : loss
        }

