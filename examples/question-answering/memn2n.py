# License needed

import numpy as np
import tensorflow as tf


class UserModel(Tower):

    @model_property
    def inference(self):

        def position_encoding(sentence_size, embedding_size):
            """
            Position Encoding described in section 4.1 [1]
            """
            encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
            ls = sentence_size+1
            le = embedding_size+1
            for i in range(1, le):
                for j in range(1, ls):
                    encoding[i-1, j-1] = (i - (le-1)/2) * (j - (ls-1)/2)
            encoding = 1 + 4 * encoding / embedding_size / sentence_size
            return np.transpose(encoding)

        def memn2n(x, embeddings, weights, encoding, hops):
            """
            Create model
            """

            with tf.variable_scope("memn2n"):
                # x has shape [batch_size, story_size, sentence_size, 2]
                # unpack along last dimension to extract stories and questions
                stories, questions = tf.unpack(x, axis=3)

                self.summaries.append(tf.histogram_summary("stories_hist", stories))
                self.summaries.append(tf.histogram_summary("questions_hist", questions))

                # assume single sentence in question
                questions = questions[:, 0, :]

                self.summaries.append(tf.histogram_summary("question_hist", questions))

                q_emb = tf.nn.embedding_lookup(embeddings['B'], questions, name='q_emb')
                u_0 = tf.reduce_sum(q_emb * encoding, 1)
                u = [u_0]
                for _ in xrange(hops):
                    m_emb = tf.nn.embedding_lookup(embeddings['A'], stories, name='m_emb')
                    m = tf.reduce_sum(m_emb * encoding, 2) + weights['TA']
                    # hack to get around no reduce_dot
                    u_temp = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                    dotted = tf.reduce_sum(m * u_temp, 2)

                    # Calculate probabilities
                    probs = tf.nn.softmax(dotted)

                    probs_temp = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                    c_temp = tf.transpose(m, [0, 2, 1])
                    o_k = tf.reduce_sum(c_temp * probs_temp, 2)

                    u_k = tf.matmul(u[-1], weights['H']) + o_k

                    if self.is_training:
                        u_k = tf.nn.dropout(u_k, 0.75)

                    u.append(u_k)

                o = tf.matmul(u_k, weights['W'])
                if self.is_training:
                    o = tf.nn.dropout(o, 0.75)
            return o

        # configuration
        sentence_size = self.input_shape[1]
        story_size = self.input_shape[0]
        embedding_size = 25
        hops = 3
        dict_size = 40
        encoding = tf.constant(position_encoding(sentence_size, embedding_size), name="encoding")
        x = tf.to_int32(tf.reshape(self.x, shape=[-1, story_size, sentence_size, 2]), name='x_int')

        # model parameters
        initializer = tf.random_normal_initializer(stddev=0.1)
        embeddings = {
            'A': tf.get_variable('A', [dict_size, embedding_size], initializer=initializer),
            'B': tf.get_variable('B', [dict_size, embedding_size], initializer=initializer),
        }
        weights = {
            'TA': tf.get_variable('TA', [story_size, embedding_size], initializer=initializer),
            'H': tf.get_variable('H', [embedding_size, embedding_size], initializer=initializer),
            'W': tf.get_variable('W', [embedding_size, dict_size], initializer=initializer),
        }

        self._nil_vars = [embeddings['A'].name, embeddings['B'].name]

        self.summaries.append(tf.histogram_summary("A_hist", embeddings['A']))
        self.summaries.append(tf.histogram_summary("B_hist", embeddings['B']))
        self.summaries.append(tf.histogram_summary("TA_hist", weights['TA']))
        self.summaries.append(tf.histogram_summary("H_hist", weights['H']))
        self.summaries.append(tf.histogram_summary("W_hist", weights['W']))
        self.summaries.append(tf.histogram_summary("X_hist", x))

        # create model
        model = memn2n(x, embeddings, weights, encoding, hops)

        return model

    @model_property
    def loss(self):
        # label has shape [batch_size, 1, story_size, sentence_size]
        # assume single-word labels
        y = tf.to_int64(self.y[:, 0, 0, 0], name='y_int')
        self.summaries.append(tf.histogram_summary("Y_hist", y))
        loss = digits.classification_loss(self.inference, y)
        accuracy = digits.classification_accuracy(self.inference, y)
        self.summaries.append(tf.scalar_summary(accuracy.op.name, accuracy))
        return loss

    def gradientUpdate(self, grads_and_vars):
        def add_gradient_noise(t, stddev=1e-3, name=None):
            t = tf.convert_to_tensor(t, name="t")
            gn = tf.random_normal(tf.shape(t), stddev=stddev)
            return tf.add(t, gn, name=name)

        def zero_nil_slot(t, name=None):
            t = tf.convert_to_tensor(t, name="t")
            s = tf.shape(t)[1]
            z = tf.zeros(tf.pack([1, s]))
            return tf.concat(0, [z, tf.slice(t, [1, 0], [-1, -1])], name=name)
        max_grad_norm = 40.0
        grads_and_vars = [(tf.clip_by_norm(g, max_grad_norm), v) for g, v in grads_and_vars]
        grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                print("grad zero nil slot")
                g = zero_nil_slot(g)
                g = tf.Print(g, [g], message="This is g: ", first_n=10, summarize=100)
                nil_grads_and_vars.append((g, v))
            else:
                nil_grads_and_vars.append((g, v))
        return nil_grads_and_vars
