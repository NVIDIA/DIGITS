from model import Tower
from utils import model_property
import tensorflow as tf
import utils as digits


class UserModel(Tower):

    @model_property
    def inference(self):
        const = tf.constant(0.004)
        normed = tf.multiply(self.x, const)

        # The reshaping have to be done for tensorflow to get the shape right
        right_shape = tf.reshape(normed, shape=[-1, 50, 50])
        transposed = tf.transpose(right_shape, [0, 2, 1])
        squeezed = tf.reshape(transposed, shape=[-1, 2500])

        # Define weights
        weights = {
            'w1': tf.get_variable('w1', [2500, 2])
        }
        biases = {
            'b1': tf.get_variable('b1', [2])
        }

        # Linear activation
        model = tf.matmul(squeezed, weights['w1']) + biases['b1']
        tf.summary.image(model.op.name, model, max_outputs=1, collections=["Training Summary"])
        return model

    @model_property
    def loss(self):
        label = tf.reshape(self.y, shape=[-1, 2])
        model = self.inference
        loss = digits.mse_loss(model, label)
        return loss
