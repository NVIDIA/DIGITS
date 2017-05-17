from tensorflow.python.ops import rnn, rnn_cell

def build_model(params):
    n_hidden = 28
    n_classes = params['nclasses']
    n_steps = params['input_shape'][0]
    n_input = params['input_shape'][1]
    
    x = tf.reshape(params['x'], shape=[-1, params['input_shape'][0], params['input_shape'][1], params['input_shape'][2]])
    
    tf.image_summary(x.op.name, x, max_images=1, collections=[digits.GraphKeys.SUMMARIES_TRAIN])
    x = tf.squeeze(x)
    
    

    # Define weights
    weights = {
        'w1': tf.get_variable('w1', [n_hidden, params['nclasses']])
    }
    biases = {
        'b1': tf.get_variable('b1', [params['nclasses']])
    }
    
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    model = tf.matmul(outputs[-1], weights['w1']) + biases['b1']

    def loss(y):
        loss = digits.classification_loss(model, y)
        accuracy = digits.classification_accuracy(model, y)
        tf.summary.scalar(accuracy.op.name, accuracy, collections=[digits.GraphKeys.SUMMARIES_TRAIN])
        return loss

    return {
        'model' : model,
        'loss' : loss
        }
