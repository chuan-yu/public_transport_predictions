import os
import tensorflow as tf
import reader
import numpy as np
from matplotlib import pyplot as plt

class Simple_LSTM():
    def __init__(self, config):
        self.state_size = config.state_size
        self.num_layers = config.num_layers
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.time_steps = config.time_steps
        self.lr = config.lr
        self.epochs = config.num_epochs
        self._sess = tf.Session()
        self._val_loss = 0

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
        self._batchX_placeholder = tf.placeholder(tf.float32, [None, self.time_steps, self.input_size],
                                                  name="input")
        self._batchY_placeholder = tf.placeholder(tf.float32, [None, self.output_size],
                                                  name="label")
        self._batch_size = tf.placeholder(tf.int32, [], name="batch_size")

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

            lstm_cells = self._get_lstm_cells()

            # Zero initial state
            init_states = lstm_cells.zero_state(self._batch_size, dtype=tf.float32)

            states_series, self._current_state = tf.nn.dynamic_rnn(lstm_cells, self._batchX_placeholder,
                                                                         initial_state=init_states)

            last_time_step = states_series[:, -1, :]
            self._prediction_series = tf.matmul(last_time_step, self._w_out) + self._b_out

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
            # self._val_summary = tf.summary.scalar("validation_rmse", self._val_loss)

    def save_graph(self):
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./graphs', sess.graph)
        writer.close()

    def fit(self, x_train, y_train, x_val, y_val):

        writer = tf.summary.FileWriter("summaries/")
        num_batches = x_train.shape[0]

        for epoch in range(config.num_epochs):
            for i in range(num_batches):

                total_loss, train_steps, train_summ = self._sess.run(
                    [self._total_loss, self._train_step, self._train_summary],
                    feed_dict={
                        self._batchX_placeholder: x_train[i, :, :, :],
                        self._batchY_placeholder: y_train[i, :, :],
                        self._batch_size: config.train_batch_size
                    })

                writer.add_summary(train_summ, global_step=self._sess.run(self._global_step))

            # Evaluate validation loss at the end of each epoch
            val_loss_series = np.ndarray(x_val.shape[0])

            for j in range(x_val.shape[0]):
                val_loss = self._sess.run(self._total_loss,
                                          feed_dict={
                                              self._batchX_placeholder: x_val[[j], :, :],
                                              self._batchY_placeholder: y_val[[j], :],
                                              self._batch_size: 1
                                          })
                val_loss_series[j] = val_loss

            self._val_loss = self._sess.run(tf.reduce_mean(val_loss_series))

            summary = tf.Summary(value=[tf.Summary.Value(tag="validation_rmse",
                                                         simple_value=self._val_loss)])
            writer.add_summary(summary, global_step=self._sess.run(self._global_step))
            writer.flush()

            # print the errors at the end of each epoch
            print("Step", epoch, "train loss", total_loss, ", val_loss: ", self._val_loss)
            # print("Step", epoch, "train loss", total_loss)


            if (epoch + 1) % 10 == 0:
                self._saver.save(self._sess, "checkpoints/simple_lstm.ckpt", global_step=self._global_step)

        writer.close()

    def predict(self, x):

        prediction_array = np.ndarray((x.shape[0], x.shape[2]))
        for i in range(x.shape[0]):
            prediction = self._sess.run(self._prediction_series,
                                         feed_dict={
                                             self._batchX_placeholder: x[[i], :, :],
                                             self._batch_size: 1
                                         })
            prediction_array[i] = prediction

        return prediction_array

    def predict_multiple_steps(self, x, num_steps, initial_state=None):
        batchX = np.reshape(x, (1, x.shape[0], x.shape[1]))
        prediction_list = []
        for i in range(num_steps):
            predictions = self._sess.run(self._prediction_series,
                                                        feed_dict={
                                                            self._batchX_placeholder: batchX,
                                                            self._batch_size: 1
                                                        })

            prediction_list.append(predictions.reshape(predictions.shape[1]))
            batchX = np.append(batchX, predictions.reshape((1, 1, predictions.shape[1])), axis=1)
            batchX = np.delete(batchX, 0, axis=1)

        prediction_list = np.array(prediction_list)
        return prediction_list


class LSTMConfig():
    train_batch_size = 20
    state_size = 32
    num_layers = 1
    input_size = 141
    output_size = 141
    time_steps = 50
    lr = 0.01
    num_epochs = 200
    checkpoint = "checkpoints/simple_lstm.ckpt"


if __name__ == "__main__":

    config = LSTMConfig()

    # Load data
    data_path = "data/count_by_hour_with_header.csv"
    train, val, test = reader.mrt_simple_lstm_data(config.train_batch_size, config.time_steps,
                                                   data_path=data_path)
    x_train, y_train = train[0], train[1]
    x_val, y_val, = val[0], val[1]
    x_test, y_test = test[0], test[1]
    # x_full, y_full = reader.get_raw_data()

    # Build model
    lstm_model = Simple_LSTM(config)

    # Run training
    # lstm_model.fit(x_train, y_train, x_val, y_val)

    # # Make predictions
    x_input = x_test[0, :, :]
    # time_features = x_test[0, 1:, -2:]
    # # # predictions = lstm_model.predict(x_test)
    # _, initial_state = lstm_model.predict(x_val)

    predictions = lstm_model.predict_multiple_steps(x_input, y_test.shape[0])
    # predictions = lstm_model.predict(x_test)
    plt.plot(predictions[0:59, 0], label="predictions")
    plt.plot(y_test[0:59, 0], label="true values")
    plt.legend(loc='upper right')
    plt.show()
    print("done")



