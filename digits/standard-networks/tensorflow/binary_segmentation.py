# Tensorflow Triangle binary segmentation model using TensorFlow-Slim

def build_model(params):
    _x = tf.reshape(params['x'], shape=[-1, params['input_shape'][0], params['input_shape'][1], params['input_shape'][2]])
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected],  
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        weights_regularizer=slim.l2_regularizer(0.0005) ):

        model = slim.conv2d(_x, 32, [3, 3], padding='SAME', scope='conv1') # 1*H*W -> 32*H*W
        model = slim.conv2d(model, 1024, [16, 16], padding='VALID', scope='conv2', stride=16) # 32*H*W -> 1024*H/16*W/16
        model = slim.conv2d_transpose(model, params['input_shape'][2], [16, 16], stride=16, padding='VALID', activation_fn=None, scope='deconv_1')

    def loss(y):
        y = tf.reshape(y, shape=[-1, params['input_shape'][0], params['input_shape'][1], params['input_shape'][2]])
        # For a fancy tensorboard summary: put the input, label and model side by side (sbs) for a fancy image summary:
        # sbs = tf.concat(2, [_x, y, model])
        # tf.image_summary(sbs.op.name, sbs, max_images=3, collections=[digits.GraphKeys.SUMMARIES_TRAIN])
        return digits.mse_loss(model, y)

    return {
        'model' : model,
        'loss' : loss
        }
