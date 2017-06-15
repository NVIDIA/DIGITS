class UserModel(Tower):

    all_inception_settings = {
        '3b': [[64], [96, 128], [16, 32], [32]],
        '3c': [[128], [128, 192], [32, 96], [64]],
        '4b': [[198], [96, 128], [16, 48], [64]],
        '4c': [[160], [112, 224], [24, 64], [64]],
        '4d': [[128], [128, 256], [24, 64], [64]],
        '4e': [[112], [144, 288], [32, 64], [64]],
        '4f': [[256], [160, 320], [32, 128], [128]],
        '5b': [[256], [160, 320], [32, 128], [128]],
        '5c': [[384], [192, 384], [48, 128], [128]]
    }

    @model_property
    def inference(self):
        # rescale to proper form, really we expect 256 x 256 x 1 in HWC form
        model = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        conv_7x7_2s_vars = create_vars([7, 7, self.input_shape[2], 64], 'conv_7x7_2s')
        model = conv_layer_with_relu(model, conv_7x7_2s_vars.weight, conv_7x7_2s_vars.bias, 7)

        model = max_pool(model, 3, 2)
        model = tf.nn.local_response_normalization(model)

        conv_1x1_vs_vars = create_vars([1, 1, 64, 64], 'conv_1x1_vs')
        model = conv_layer_with_relu(model, conv_1x1_vs_vars.weight, conv_1x1_vs_vars.bias, 1)

        conv_3x3_1s_vars = create_vars([3, 3, 64, 192], 'conv_3x3_1s')
        model = conv_layer_with_relu(model, conv_3x3_1s_vars.weight, conv_3x3_1s_vars.bias, 3)

        model = tf.nn.local_response_normalization(model)

        model = max_pool(model, 3, 2)

        inception_settings_3b = InceptionSettings(192, all_inception_settings['3b'])
        model = inception(model, inception_settings_3b, '3b')

        inception_settings_3c = InceptionSettings(256, all_inception_settings['3c'])
        model = inception(model, inception_settings_3c, '3c')

        model = max_pool(model, 3, 2)

        inception_settings_4b = InceptionSettings(480, all_inception_settings['4b'])
        model = inception(model, inception_settings_4b, '4b')

        inception_settings_4c = InceptionSettings(512, all_inception_settings['4c'])
        model = inception(model, inception_settings_4c, '4c')

        inception_settings_4d = InceptionSettings(512, all_inception_settings['4d'])
        model = inception(model, inception_settings_4d, '4d')

        inception_settings_4e = InceptionSettings(512, all_inception_settings['4e'])
        model = inception(model, inception_settings_4e, '4e')

        inception_settings_4f = InceptionSettings(832, all_inception_settings['4f'])
        model = inception(model, inception_settings_4f, '4f')

        model = max_pool(model, 3, 2)

        inception_settings_5b = InceptionSettings(832, all_inception_settings['5b'])
        model = inception(model, inception_settings_5b, '5b')

        inception_settings_5c = InceptionSettings(1024, all_inception_settings['5c'])
        model = inception(model, inception_settings_5c, '5c')

    @model_property
    def loss(self):

    def inception(model, inception_setting, layer_name):
        variables = create_inception_variables(inception_setting, layer_name)
        conv_1x1 = conv_layer_with_relu(model, variables.weights['conv_1x1_1'], variables.biases['conv_1x1_1'], 1)

        conv_3x3 = conv_layer_with_relu(model, variables.weights['conv_1x1_2'], variables.biases['conv_1x1_2'], 1)
        conv_3x3 = conv_layer_with_relu(conv_3x3, variables.weights['conv_3x3'], variables.biases['conv_3x3'], 3)

        conv_5x5 = conv_layer_with_relu(model, variables.weights['conv_1x1_3'], variables.biases['conv_1x1_3'], 1)
        conv_5x5 = conv_layer_with_relu(conv_5x5, variables.weights['conv_5x5'], variables.biases['conv_5x5'], 5)

        conv_pool = max_pool(model, 3, 1)
        conv_pool = conv_layer_with_relu(conv_pool, variables.weights['conv_pool'], variables.biases['conv_pool'], 1)

        final_model = tf.concat([conv_1x1, conv_3x3, conv_5x5, conv_pool], 3)

        return final_model
        
    def create_inception_variables(inception_setting, layer_name):
        model_dim = inception_setting.model_dim
        conv_1x1_1_var = create_vars([1, 1, model_dim, inception_setting.conv_1x1_1_layers], layer_name + '-' + 'conv_1x1_1')
        conv_1x1_2_var = create_vars([1, 1, model_dim, inception_setting.conv_1x1_2_layers], layer_name + '-' + 'conv_1x1_2')
        conv_1x1_3_var = create_vars([1, 1, model_dim, inception_setting.conv_1x1_3_layers], layer_name + '-' + 'conv_1x1_3')
        conv_3x3_var = create_vars([3, 3, inception_setting.conv_1x1_2_layers, inception_setting.conv_3x3_layers], layer_name + '-' + 'conv_3x3')
        conv_5x5_var = create_vars([5, 5, inception_setting.conv_5x5_layers, inception_setting.conv_5x5_layers], layer_name + '-' + 'conv_5x5'),
        conv_pool_var = create_vars([1, 1, model_dim, inception_setting.conv_pool_layers], layer_name + '-' + 'conv_pool')

        weights = {
            'conv_1x1_1': conv_1x1_1_var.weight
            'conv_1x1_2': conv_1x1_2_var.weight
            'conv_1x1_3': conv_1x1_3_var.weight
            'conv_3x3': conv_3x3_var.weight
            'conv_5x5': conv_5x5_var.weight
            'conv_pool': conv_pool_var.weight
        }

        biases = {
            'conv_1x1_1': conv_1x1_1_var.bias
            'conv_1x1_2': conv_1x1_2_var.bias
            'conv_1x1_3': conv_1x1_3_var.bias
            'conv_3x3': conv_3x3_var.bias
            'conv_5x5': conv_5x5_var.bias
            'conv_pool': conv_pool_var.bias
        }

        return {weights: weights, biases: biases}

    def conv_layer_with_relu(model, weights, biases, stride_size, padding='SAME'):
        new_model = tf.nn.conv2d(model, weights, strides=[1, stride_size, stride_size, 1], padding=padding)
        new_model = tf.nn.bias_add(new_model, biases)
        new_model = tf.nn.relu(new_model)
        return new_model

    def max_pool(model, kernal_size, stride_size, padding='SAME'):
        new_model = tf.nn.max_pool(model, ksize=[1, kernal_size, kernal_size, 1], strides=[1, stride_size, stride_size, 1], padding=padding)
        return new_model

    def avg_pool(model, kernal_size, stride_size, padding='SAME'):
        new_model = tf.nn.avg_pool(model, ksize=[1, kernal_size, kernal_size, 1], strides=[1, stride_size, stride_size, 1], padding=padding)
        return new_model

    def create_vars(size, name):
        weight = create_weight(size, name + '_W')
        bias = create_bias(size[3], name + '_b')
        return {weight: weight, bias: bias}

    def create_weight(size, name):
        weight = tf.get_variable(name, size=size, initializer=tf.contrib.layers.xaiver_initializer())
        return weight

    def create_bias(size, name):
        bias = tf.get_variable(name, size=[size], initializer=tf.constant_initializer(0.0))
        return bias

class InceptionSettings():
    
    def __init__(self, model_dim, inception_settings):
        self.model_dim = model_dim
        self.conv_1x1_1_layers = inception_settings[0][0]
        self.conv_1x1_2_layers = inception_settings[1][0]
        self.conv_1x1_3_layers = inception_settings[2][0]
        self.conv_3x3_layers = inception_settings[1][1]
        self.conv_5x5_layers = inception_settings[2][1]
        self.conv_pool_layers = inception_settings[3][0]