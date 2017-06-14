class UserModel(Tower):

    @model_property
    def inference(self):
        # rescale to proper form, really we expect 256 x 256 x 1 in HWC form
        x = tf.reshape(self.x, shape=[-1, self.input_shape[0], self.input_shape[1], self.input_shape[2]])

    @model_property
    def loss(self):

    def inception(model, inception_setting, layer_name):
        
        
    def create_inception_variables(inception_setting, layer_name):
        model_dim = inception_setting.model_dim
        weights = {
            'conv_1x1_1': create_weight([1, 1, model_dim, inception_setting.conv_1x1_1_layers], layer_name + '-' + 'conv_1x1_1'),

        }

    def conv_layer_with_relu(model, weights, biases, stride_size, padding="SAME"):
        new_model = tf.nn.conv2d(model, weights, strides=[1, stride_size, stride_size, 1], padding=padding)
        new_model = tf.nn.bias_add(new_model, biases)
        new_model = tf.nn.relu(new_model)
        return new_model

    def max_pool(model, kernal_size, stride_size, padding="SAME"):
        new_model = tf.nn.max_pool(model, ksize=[1, kernal_size, kernal_size, 1], strides=[1, stride_size, stride_size, 1], padding=padding)
        return new_model

    def avg_pool(model, kernal_size, stride_size, padding="SAME"):
        new_model = tf.nn.avg_pool(model, ksize=[1, kernal_size, kernal_size, 1], strides=[1, stride_size, stride_size, 1], padding=padding)
        return new_model

    def create_weight(size, name):
        weight = tf.get_variable(name, size=size, initializer=tf.contrib.layers.xaiver_initializer())
        return weight

    def create_bias(size, name):
        bias = tf.get_variable(name, size=size, initializer=tf.constant_initializer(0.0))
        return bias

class InceptionSettings():
    
    def __init__(self, model_dim, conv_1x1_1_layers, conv_1x1_2_layers, 
                 conv_1x1_3_layers, conv_3x3_layers, conv_5x5_layers, pool_layers):
        self.model_dim = model_dim
        self.conv_1x1_1_layers = conv_1x1_1_layers
        self.conv_1x1_2_layers = conv_1x1_2_layers
        self.conv_1x1_3_layers = conv_1x1_3_layers
        self.conv_3x3_layers = conv_3x3_layers
        self.conv_5x5_layers = conv_5x5_layers
        self.pool_layers = pool_layers