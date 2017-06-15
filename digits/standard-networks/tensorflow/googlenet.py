class UserModel(Tower):

    all_inception_settings = {
        '3b': [[64], [96, 128], [16, 32], [32]],
        '3c': [[128], [128, 192], [32, 96], [64]],
        '4b': [[192], [96, 208], [16, 48], [64]],
        '4c': [[160], [112, 224], [24, 64], [64]],
        '4d': [[128], [128, 256], [24, 64], [64]],
        '4e': [[112], [144, 288], [32, 64], [64]],
        '4f': [[256], [160, 320], [32, 128], [128]],
        '5b': [[256], [160, 320], [32, 128], [128]],
        '5c': [[384], [192, 384], [48, 128], [128]]
    }

    @model_property
    def inference(self):
        # rescale to proper form, really we expect 224 x 224 x 1 in HWC form
        model = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])

        conv_7x7_2s_weight, conv_7x7_2s_bias = self.create_conv_vars([7, 7, self.input_shape[2], 64], 'conv_7x7_2s')
        model = self.conv_layer_with_relu(model, conv_7x7_2s_weight, conv_7x7_2s_bias, 2)

        model = self.max_pool(model, 3, 2)

        # according to Tim, this is slow as hell
        model = tf.nn.local_response_normalization(model)

        conv_1x1_vs_weight, conv_1x1_vs_bias = self.create_conv_vars([1, 1, 64, 64], 'conv_1x1_vs')
        model = self.conv_layer_with_relu(model, conv_1x1_vs_weight, conv_1x1_vs_bias, 1, 'VALID')

        conv_3x3_1s_weight, conv_3x3_1s_bias = self.create_conv_vars([3, 3, 64, 192], 'conv_3x3_1s')
        model = self.conv_layer_with_relu(model, conv_3x3_1s_weight, conv_3x3_1s_bias, 1)

        model = tf.nn.local_response_normalization(model)

        model = self.max_pool(model, 3, 2)

        inception_settings_3b = InceptionSettings(192, UserModel.all_inception_settings['3b'])
        model = self.inception(model, inception_settings_3b, '3b')

        inception_settings_3c = InceptionSettings(256, UserModel.all_inception_settings['3c'])
        model = self.inception(model, inception_settings_3c, '3c')

        model = self.max_pool(model, 3, 2)

        inception_settings_4b = InceptionSettings(480, UserModel.all_inception_settings['4b'])
        model = self.inception(model, inception_settings_4b, '4b')

        # first auxiliary branch for making training faster
        aux_branch_1 = self.auxiliary_classifier(model, 512, "aux_1")

        inception_settings_4c = InceptionSettings(512, UserModel.all_inception_settings['4c'])
        model = self.inception(model, inception_settings_4c, '4c')

        inception_settings_4d = InceptionSettings(512, UserModel.all_inception_settings['4d'])
        model = self.inception(model, inception_settings_4d, '4d')

        inception_settings_4e = InceptionSettings(512, UserModel.all_inception_settings['4e'])
        model = self.inception(model, inception_settings_4e, '4e')

        # second auxiliary branch for making training faster
        aux_branch_2 = self.auxiliary_classifier(model, 528, "aux_2")

        inception_settings_4f = InceptionSettings(528, UserModel.all_inception_settings['4f'])
        model = self.inception(model, inception_settings_4f, '4f')

        model = self.max_pool(model, 3, 2)

        inception_settings_5b = InceptionSettings(832, UserModel.all_inception_settings['5b'])
        model = self.inception(model, inception_settings_5b, '5b')

        inception_settings_5c = InceptionSettings(832, UserModel.all_inception_settings['5c'])
        model = self.inception(model, inception_settings_5c, '5c')
        
        model = self.avg_pool(model, 7, 1, 'VALID')

        fc_weight, fc_bias = self.create_fc_vars([1024, self.nclasses], 'fc')
        model = self.fully_connect(model, fc_weight, fc_bias)

#        model = tf.nn.softmax(model)

        if self.is_training:
            return tf.add(tf.add(aux_branch_1, aux_branch_2), model)
        else:
            return model

    @model_property
    def loss(self):
        model = self.inference
        loss = digits.classification_loss(model, self.y)
        accuracy = digits.classification_accuracy(model, self.y)
        self.summaries.append(tf.summary.scalar(accuracy.op.name, accuracy))
        return loss


    def inception(self, model, inception_setting, layer_name):
        weights, biases = self.create_inception_variables(inception_setting, layer_name)
        conv_1x1 = self.conv_layer_with_relu(model, weights['conv_1x1_1'], biases['conv_1x1_1'], 1)

        conv_3x3 = self.conv_layer_with_relu(model, weights['conv_1x1_2'], biases['conv_1x1_2'], 1)
        conv_3x3 = self.conv_layer_with_relu(conv_3x3, weights['conv_3x3'], biases['conv_3x3'], 1)

        conv_5x5 = self.conv_layer_with_relu(model, weights['conv_1x1_3'], biases['conv_1x1_3'], 1)
        conv_5x5 = self.conv_layer_with_relu(conv_5x5, weights['conv_5x5'], biases['conv_5x5'], 1)

        conv_pool = self.max_pool(model, 3, 1)
        conv_pool = self.conv_layer_with_relu(conv_pool, weights['conv_pool'], biases['conv_pool'], 1)

        final_model = tf.concat([conv_1x1, conv_3x3, conv_5x5, conv_pool], 3)

        return final_model
        
    def create_inception_variables(self, inception_setting, layer_name):
        model_dim = inception_setting.model_dim
        conv_1x1_1_weight, conv_1x1_1_bias = self.create_conv_vars([1, 1, model_dim, inception_setting.conv_1x1_1_layers], layer_name + '-conv_1x1_1')
        conv_1x1_2_weight, conv_1x1_2_bias = self.create_conv_vars([1, 1, model_dim, inception_setting.conv_1x1_2_layers], layer_name + '-conv_1x1_2')
        conv_1x1_3_weight, conv_1x1_3_bias = self.create_conv_vars([1, 1, model_dim, inception_setting.conv_1x1_3_layers], layer_name + '-conv_1x1_3')
        conv_3x3_weight, conv_3x3_bias = self.create_conv_vars([3, 3, inception_setting.conv_1x1_2_layers, inception_setting.conv_3x3_layers], layer_name + '-conv_3x3')
        conv_5x5_weight, conv_5x5_bias = self.create_conv_vars([5, 5, inception_setting.conv_1x1_3_layers, inception_setting.conv_5x5_layers], layer_name + '-conv_5x5')
        conv_pool_weight, conv_pool_bias = self.create_conv_vars([1, 1, model_dim, inception_setting.conv_pool_layers], layer_name + '-conv_pool')

        weights = {
            'conv_1x1_1': conv_1x1_1_weight,
            'conv_1x1_2': conv_1x1_2_weight,
            'conv_1x1_3': conv_1x1_3_weight,
            'conv_3x3': conv_3x3_weight,
            'conv_5x5': conv_5x5_weight,
            'conv_pool': conv_pool_weight
        }

        biases = {
            'conv_1x1_1': conv_1x1_1_bias,
            'conv_1x1_2': conv_1x1_2_bias,
            'conv_1x1_3': conv_1x1_3_bias,
            'conv_3x3': conv_3x3_bias,
            'conv_5x5': conv_5x5_bias,
            'conv_pool': conv_pool_bias
        }

        return weights, biases

    def auxiliary_classifier(self, model, input_size, name):
        aux_classifier = self.avg_pool(model, 5, 3, 'VALID')

        conv_weight, conv_bias = self.create_conv_vars([1, 1, input_size, input_size], name + '-conv_1x1')
        aux_classifier = self.conv_layer_with_relu(aux_classifier, conv_weight, conv_bias, 1)

        fc_weight, fc_bias = self.create_fc_vars([4*4*input_size, self.nclasses], name + '-fc')
        aux_classifier = self.fully_connect(aux_classifier, fc_weight, fc_bias)

        aux_classifier = tf.nn.dropout(aux_classifier, 0.7)

#        aux_classifier = tf.nn.softmax(aux_classifier)

        return aux_classifier

    def conv_layer_with_relu(self, model, weights, biases, stride_size, padding='SAME'):
        new_model = tf.nn.conv2d(model, weights, strides=[1, stride_size, stride_size, 1], padding=padding)
        new_model = tf.nn.bias_add(new_model, biases)
        new_model = tf.nn.relu(new_model)
        return new_model

    def max_pool(self, model, kernal_size, stride_size, padding='SAME'):
        new_model = tf.nn.max_pool(model, ksize=[1, kernal_size, kernal_size, 1], strides=[1, stride_size, stride_size, 1], padding=padding)
        return new_model

    def avg_pool(self, model, kernal_size, stride_size, padding='SAME'):
        new_model = tf.nn.avg_pool(model, ksize=[1, kernal_size, kernal_size, 1], strides=[1, stride_size, stride_size, 1], padding=padding)
        return new_model

    def fully_connect(self, model, weights, biases):
        fc_model = tf.reshape(model, [-1, weights.get_shape().as_list()[0]])
        fc_model = tf.matmul(fc_model, weights)
        fc_model = tf.add(fc_model, biases)
        fc_model = tf.nn.relu(fc_model)
        return fc_model

    def create_conv_vars(self, size, name):
        weight = self.create_weight(size, name + '_W')
        bias = self.create_bias(size[3], name + '_b')
        return weight, bias

    def create_fc_vars(self, size, name):
        weight = self.create_weight(size, name + '_W')
        bias = self.create_bias(size[1], name + '_b')
        return weight, bias

    def create_weight(self, size, name):
        weight = tf.get_variable(name, size, initializer=tf.contrib.layers.xavier_initializer())
        return weight

    def create_bias(self, size, name):
        bias = tf.get_variable(name, [size], initializer=tf.constant_initializer(0.0))
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