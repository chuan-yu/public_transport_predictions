import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import reader

class Seq2Seq():
    def __init__(self, config, inference=False):
        tf.reset_default_graph()

        self.state_size = config.state_size
        self.num_layers = config.num_layers
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.input_time_steps = config.input_time_steps
        self.output_time_steps = config.output_time_steps
        self.lr = config.lr
        self.epochs = config.num_epochs
        self._sess = tf.Session()

        self._build_graph(inference)
        self._define_loss()
        self._define_optimizer()
        # self._save_graph()

        init = tf.global_variables_initializer()
        self._sess.run(init)

        self._saver = tf.train.Saver()

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.checkpoint))
        if ckpt and ckpt.model_checkpoint_path:
            print("restoring checkpoints", ckpt.model_checkpoint_path)
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)

    def _build_graph(self, inference=False):

        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

        self._w_out = tf.get_variable("output_weights", dtype=tf.float32,
                                initializer=tf.truncated_normal([self.state_size, self.output_size]))
        self._b_out = tf.get_variable("output_bias", dtype=tf.float32,
                                initializer=tf.truncated_normal([1, self.output_size]))

        self._encoder_inp = tf.placeholder(tf.float32, [self.input_time_steps, None, self.input_size],
                                     name="encoder_input")
        _, enc_state = self._build_encoder(self._encoder_inp)

        self._decoder_target = tf.placeholder(tf.float32, [self.output_time_steps, None, self.output_size],
                                     name="decoder_target")
        go_signal = tf.zeros_like(tf.expand_dims(self._decoder_target[0], 0), dtype=tf.float32, name="GO")

        dec_inp = tf.concat([go_signal, self._decoder_target], 0)
        dec_output, _ = self._build_decoder(dec_inp, enc_state, inference)

        self._model_outputs = [tf.matmul(out, self._w_out) + self._b_out for out in dec_output]

    def _define_loss(self):
        with tf.variable_scope("Loss"):
            self._loss = tf.reduce_sum(tf.sqrt(tf.reduce_mean(
                tf.square(tf.subtract(self._model_outputs, self._decoder_target)), 1)))

    def _define_optimizer(self):
        with tf.variable_scope("Training"):
            self._train_step = tf.train.AdadeltaOptimizer(self.lr).minimize(self._loss, global_step=self._global_step)

    def _get_lstm_cells(self):
        cell = tf.contrib.rnn.BasicLSTMCell(self.state_size, state_is_tuple=True)
        cells = tf.contrib.rnn.MultiRNNCell([cell] * self.num_layers)
        return cells

    def _build_encoder(self, enc_inp):
        with tf.variable_scope("Encoder"):

            encoder_cells = self._get_lstm_cells()
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cells, enc_inp,
                                                               time_major=True, dtype=tf.float32)

        return encoder_outputs, encoder_state

    def _build_decoder(self, dec_inp, init_state, inference=False):
        with tf.variable_scope("Decoder") as dec_scope:
            decoder_cells = self._get_lstm_cells()
            state = init_state
            outputs = []
            prev_output = None
            for i in range(self.output_time_steps):
                inp = dec_inp[i]
                if inference and prev_output is not None:
                    inp = tf.matmul(prev_output, self._w_out) + self._b_out
                if i > 0:
                    dec_scope.reuse_variables()
                output, state = decoder_cells(inp, state)
                outputs.append(output)
                prev_output = output

        return outputs, state

    def _save_graph(self):
        writer = tf.summary.FileWriter('./graphs', self._sess.graph)
        writer.close()

    def fit(self, x_train, y_train, x_val, y_val):

        num_batches = x_train.shape[0]

        for epoch in range(config.num_epochs):
            for i in range(num_batches):
                train_loss, train_steps = self._sess.run(
                    [self._loss, self._train_step],
                    feed_dict={
                        self._encoder_inp: x_train[i],
                        self._decoder_target: y_train[i]
                    }
                )

            print("step", epoch, "train_loss: ", train_loss)

            if (epoch + 1) % 100 == 0:
                self._saver.save(self._sess, config.checkpoint, global_step=self._global_step)

    def predict(self, x, y):
        predictions = []
        for i in range(x.shape[0]):
            prediction = self._sess.run(self._model_outputs,
                                        feed_dict={
                                            self._encoder_inp: x[i],
                                            self._decoder_target: y[i]
                                        })
            predictions.append(prediction)

        predictions = np.array(predictions)
        return predictions


class LSTMConfig():
    train_batch_size = 10

    state_size = 32
    num_layers = 1
    input_size = 1
    output_size = 1
    input_time_steps = 50
    output_time_steps = 20
    lr = 0.01
    num_epochs = 5000
    checkpoint = "checkpoints/seq2seq/seq2seq.ckpt"

if __name__ == "__main__":

    config = LSTMConfig()

    seq2seq_model = Seq2Seq(config, inference=False)

    # Load data
    data_path = "data/count_by_hour_with_header.csv"
    data_scaled = reader.get_scaled_mrt_data(data_path, [0])
    train, val, test = reader.produce_seq2seq_data(data_scaled, config.train_batch_size, config.input_time_steps,
                                                   config.output_time_steps)

    # sin_input = np.linspace(0, 120, 420)
    # sin_data = reader.generate_sin_signal(sin_input)
    # train, val, test = reader.produce_seq2seq_data(sin_data, config.train_batch_size,
    #                                                config.input_time_steps, config.output_time_steps)
    x_train, y_train = train[0], train[1]
    x_val, y_val, = val[0], val[1]
    x_test, y_test = test[0], test[1]

    seq2seq_model.fit(x_train, y_train, x_val, y_val)

    seq2seq_model = Seq2Seq(config, inference=True)
    predictions = seq2seq_model.predict(x_test, y_test)
    plt.plot(predictions.flatten(), label="predictions")
    plt.plot(y_test.flatten(), label="true values")
    plt.legend(loc="upper right")
    plt.show()






