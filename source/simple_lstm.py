import os
import tensorflow as tf
import reader
import numpy as np
from matplotlib import pyplot as plt

class Simple_LSTM():
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.state_size = config.state_size
        self.num_layers = config.num_layers
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.time_steps = config.time_steps
        self.lr = config.lr
        self.epochs = config.num_epochs
        self._sess = tf.Session()

        self._create_placeholders()
        self._create_variables()
        self._build_lstm()
        self._define_loss()
        self._define_optimizer()
        self._define_summaries()

        self._saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        self._sess.run(init)

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(config.checkpoint))
        if ckpt and ckpt.model_checkpoint_path:
            print("have checkpoints")
            print(ckpt.model_checkpoint_path)
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)

    def _create_placeholders(self):
        self._batchX_placeholder = tf.placeholder(tf.float32, [None, None, self.input_size],
                                                  name="input")
        self._batchY_placeholder = tf.placeholder(tf.float32, [None, None, self.output_size],
                                                  name="label")
        self._initial_state = tf.placeholder(tf.float32, [self.num_layers, 2, self.batch_size, self.state_size])

    def _create_variables(self):
        self._w_out = tf.get_variable("output_weights", dtype=tf.float32,
                                      initializer=tf.truncated_normal([self.state_size, self.output_size]))
        self._b_out = tf.get_variable("output_bias", dtype=tf.float32,
                                      initializer=tf.truncated_normal([1, self.output_size]))
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

    def _get_lstm_cells(self):
        cell = tf.contrib.rnn.BasicLSTMCell(self.state_size, state_is_tuple=True)
        cells = tf.contrib.rnn.MultiRNNCell([cell] * self.num_layers)
        return cells

    def _build_lstm(self):
        with tf.variable_scope("LSTM"):

            state_per_layer_list = tf.unstack(self._initial_state, axis=0)
            rnn_tuple_state = tuple(
                [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[i][0], state_per_layer_list[i][1])
                 for i in range(self.num_layers)])

            lstm_cells = self._get_lstm_cells()
            self._states_series, self._current_state = tf.nn.dynamic_rnn(lstm_cells, self._batchX_placeholder,
                                                                         initial_state=rnn_tuple_state)

            self._states_series = tf.reshape(self._states_series, [-1, self.state_size])
            output = tf.matmul(self._states_series, self._w_out) + self._b_out
            self._prediction_series = output

    def _compute_rmse(self, labels, predictions):
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(labels, predictions))))

    def _define_loss(self):
        with tf.variable_scope("Loss"):
            self._total_loss = self._compute_rmse(self._batchY_placeholder, self._prediction_series)

    def _define_optimizer(self):
        with tf.variable_scope("Training"):
            self._train_step = tf.train.AdamOptimizer(self.lr).minimize(self._total_loss, global_step=self._global_step)

    def _define_summaries(self):
        with tf.name_scope("summaries"):
            self._train_summary = tf.summary.scalar("train_rmse", self._total_loss)
            self._val_summary = tf.summary.scalar("validation_rmse", self._total_loss)

    def save_graph(self):
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./graphs', sess.graph)
        writer.close()

    def fit(self, x_train, y_train, x_val, y_val):

        writer = tf.summary.FileWriter("summaries/")
        num_windows = x_train.shape[1] // self.time_steps
        for epoch in range(config.num_epochs):

            current_state = np.zeros((self.num_layers, 2, self.batch_size, self.state_size))

            for i in range(num_windows):
                start_idx = i * self.time_steps
                end_idx = start_idx + self.time_steps
                batchX = x_train[:, start_idx:end_idx, :]
                batchY = y_train[:, start_idx:end_idx, :]

                total_loss, train_steps, current_state, train_summ = self._sess.run(
                    [self._total_loss, self._train_step, self._current_state, self._train_summary],
                    feed_dict={
                        self._batchX_placeholder: batchX,
                        self._batchY_placeholder: batchY,
                        self._initial_state: current_state
                    })
                writer.add_summary(train_summ, global_step=self._sess.run(self._global_step))

            # Evaluate validation loss at the end of each epoch
            val_loss, val_summ = self._sess.run([self._total_loss, self._val_summary],
                                                feed_dict={
                                                    self._batchX_placeholder: x_val,
                                                    self._batchY_placeholder: y_val,
                                                    self._initial_state:
                                                        np.zeros((self.num_layers, 2, self.batch_size, self.state_size))

                                                })
            writer.add_summary(val_summ, global_step=self._sess.run(self._global_step))

            # print the errors at the end of each epoch
            print("Step", epoch, "train loss", total_loss, ", val_loss: ", val_loss)

            if (epoch + 1) % 10 == 0:
                self._saver.save(self._sess, "checkpoints/simple_lstm.ckpt", global_step=self._global_step)

        writer.close()

    def predict(self, x, initial_state=None):
        if initial_state == None:
            init_state = np.zeros((self.num_layers, 2, self.batch_size, self.state_size))
        else:
            init_state = initial_state

        predictions, last_state = self._sess.run([self._prediction_series, self._current_state],
                                     feed_dict={
                                         self._batchX_placeholder: x,
                                         self._initial_state: init_state
                                     })

        return predictions, last_state

    def predict_multiple_steps(self, x, time_features, num_steps, initial_state=None):

        batchX = np.reshape(x, (1, 1, x.shape[0]))

        if initial_state==None:
            current_state = np.zeros((self.num_layers, 2, self.batch_size, self.state_size))
        else:
            current_state = initial_state

        prediction_series = np.ndarray((num_steps, x.shape[0] - time_features.shape[1]))
        batchX_series = []
        for i in range(num_steps):
            predictions, current_state = self._sess.run([self._prediction_series, self._current_state],
                                                        feed_dict={
                                                            self._batchX_placeholder: batchX,
                                                            self._initial_state: current_state
                                                        })

            prediction_series[i] = predictions
            predictions = np.append(predictions, time_features[[i], :], axis=1)
            batchX = predictions.reshape((1, 1, predictions.shape[1]))
            batchX_series.append(batchX)


        batchX_series = np.array(batchX_series)
        batchX_series = batchX_series.reshape((batchX_series.shape[0], batchX_series.shape[3]))
        return prediction_series


class LSTMConfig():
    batch_size = 1
    state_size = 32
    num_layers = 1
    input_size = 143
    output_size = 141
    time_steps = 50
    lr = 0.01
    num_epochs = 200
    checkpoint = "checkpoints/simple_lstm.ckpt"


if __name__ == "__main__":

    config = LSTMConfig()

    # Load data
    data_path = "data/count_by_hour_with_header.csv"
    train, val, test = reader.lstm_data_producer(config.batch_size, data_path)
    x_train, y_train = train[0], train[1]
    x_val, y_val, = val[0], val[1]
    x_test, y_test = test[0], test[1]
    # x_full, y_full = reader.get_raw_data()

    # Build model
    lstm_model = Simple_LSTM(config)

    # Run training
    # lstm_model.fit(x_train, y_train, x_val, y_val)

    # # Make predictions
    x_input = x_test[0, 0, :]
    time_features = x_test[0, 1:, -2:]
    # # predictions = lstm_model.predict(x_test)
    _, initial_state = lstm_model.predict(x_val)
    predictions = lstm_model.predict_multiple_steps(x_input, time_features, y_test.shape[1]-1, initial_state=initial_state)
    # predictions, _ = lstm_model.predict(x_test, initial_state)

    plt.plot(predictions[:, 0], label="predictions")
    plt.plot(y_test[0, :, 0], label="true values")
    plt.legend(loc='upper right')
    plt.show()
    print("done")



