
def build_model(params):
    # Implementation with native tf
    ninputs = params['input_shape'][0] * params['input_shape'][1] * params['input_shape'][2]
    W = tf.get_variable('W', [ninputs, 2], initializer=tf.constant_initializer(0.0))
    b = tf.get_variable('b', [2], initializer=tf.constant_initializer(0.0)),
    model = tf.reshape(params['x'], shape=[-1, ninputs])
    model = tf.add(tf.matmul(model, W), b)
    def loss(y):
        return digits.mse_loss(model, tf.reshape(y, shape=[-1, 2]))
    return {
        'model' : model,
        'loss' : loss
        }

# def build_model(params):
#     # Implementation with tf.contrib
#     ninputs = params['input_shape'][0] * params['input_shape'][1] * params['input_shape'][2]
#     model = tf.reshape(params['x'], shape=[-1, ninputs])
#     model = tf.contrib.layers.fully_connected(model, 2)
#     def loss(y):
#         return digits.mse_loss(model, tf.reshape(y, shape=[-1, 2]))
#     return {
#         'model' : model,
#         'loss' : loss
#         }

# def build_model(params):
#     # Implementation with tf-slim
#     model = tf.reshape(params['x'], shape=[-1, params['input_shape'][0], params['input_shape'][1], params['input_shape'][2]])
#     with slim.arg_scope([slim.fully_connected]):
#         model = slim.flatten(model)
#         model = slim.fully_connected(model, 2, activation_fn=None, scope='fc')
#     def loss(y):
#         return digits.mse_loss(model, tf.reshape(y, shape=[-1, 2]))
#     return {
#         'model' : model,
#         'loss' : loss
#         }
