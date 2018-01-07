import os
import tensorflow as tf
import reader
import numpy as np
from matplotlib import pyplot as plt

class Simple_LSTM():
    ''' Simple LSTM Model
    '''

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
        '''Train the model
        :param x_train: shape [num_batches, batch_size, time_steps, feature_len]
        :param y_train: shape [num_batches, batch_size, feature len]
        :param x_val: shape [num_val_samples, time_steps, feature_len], batch_size is always 1
        :param y_val: shape [num_val_samples, feature_len], batch_size is always 1, output time step is 1
        '''

        writer = tf.summary.FileWriter("summaries/")
        num_batches = x_train.shape[0]

        for epoch in range(config.num_epochs):
            for i in range(num_batches):
                x = x_train[i]
                y = y_train[i]
                train_loss, train_steps, train_summ = self._sess.run(
                    [self._total_loss, self._train_step, self._train_summary],
                    feed_dict={
                        self._batchX_placeholder: x,
                        self._batchY_placeholder: y,
                        self._batch_size: config.train_batch_size
                    })

                writer.add_summary(train_summ, global_step=self._sess.run(self._global_step))

            # Evaluate validation loss at the end of each epoch
            val_predictions = np.ndarray((y_val.shape))

            for j in range(x_val.shape[0]):
                predictions = self._sess.run(self._prediction_series,
                                          feed_dict={
                                              self._batchX_placeholder: x_val[[j]],
                                              self._batchY_placeholder: y_val[[j]],
                                              self._batch_size: 1
                                          })
                val_predictions[j] = predictions

            val_predictions = np.array(val_predictions)

            val_loss = self._sess.run(self._compute_rmse(y_val, val_predictions))
            summary = tf.Summary(value=[tf.Summary.Value(tag="validation_rmse",
                                                         simple_value=val_loss)])
            writer.add_summary(summary, global_step=self._sess.run(self._global_step))
            writer.flush()

            # print the errors at the end of each epoch
            print("Step", epoch, "train loss", train_loss, ", val_loss: ", val_loss)
            # print("Step", epoch, "train loss", total_loss)


            if (epoch + 1) % 100 == 0:
                self._saver.save(self._sess, config.checkpoint, global_step=self._global_step)

        writer.close()

    def predict(self, x):
        '''Make one-step predictions
        :param x: shape [num_samples, time_step, feature_len], batch_size is always 1
        '''
        prediction_array = []
        for i in range(x.shape[0]):
            prediction = self._sess.run(self._prediction_series,
                                         feed_dict={
                                             self._batchX_placeholder: x[[i], :, :],
                                             self._batch_size: 1
                                         })
            prediction_array.append(prediction)

        prediction_array = np.array(prediction_array)

        return prediction_array

    def predict_multiple_steps(self, x, num_steps):
        '''Make multipel-steps predictions
        :param x: shape [time_step, feature_len], only input one sample
        :param num_steps: the number of feature steps to predict
        :return: None
        '''

        batchX = np.reshape(x, (1, x.shape[0], x.shape[1]))
        prediction_list = []
        for j in range(num_steps):
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
    train_batch_size = 30
    state_size = 30
    num_layers = 1
    input_size = 1
    output_size = 1
    time_steps = 50
    lr = 0.0001
    num_epochs = 200
    checkpoint = "checkpoints/lstm_without_time_features/simple_lstm.ckpt"


if __name__ == "__main__":

    config = LSTMConfig()

    # Build model
    lstm_model = Simple_LSTM(config)

    #########
    ## MRT
    #########

    # Load data
    data_path = "data/count_by_hour_with_header.csv"
    data_scaled = reader.get_scaled_mrt_data(data_path, [0])
    # train, val, test = reader.mrt_simple_lstm_data(data_scaled, config.train_batch_size, config.time_steps)
    train, val, test = reader.produce_seq2seq_data(data_scaled, config.train_batch_size, config.time_steps,
                                                   output_seq_len=1)
    x_train, y_train = train[0], train[1]
    x_val, y_val, = val[0], val[1]
    x_test, y_test = test[0], test[1]

    x_val = np.squeeze(x_val, axis=1)
    x_test = np.squeeze(x_test, axis=1)
    y_train = np.squeeze(y_train, axis=2)
    y_val = np.squeeze(y_val, axis=(1, 2))
    y_test = np.squeeze(y_test, axis=(1, 2))
    # y_val = np.reshape(y_val, (y_val.shape[0], y_val.shape[3]))
    # y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[3]))


    # Run training
    lstm_model.fit(x_train, y_train, x_val, y_val)

    # Make 1-step predictions
    # predictions = lstm_model.predict(x_test[:, 0, :, :])
    # plt.plot(predictions[:, 0], label="predictions")
    # y_true = y_test.reshape((y_test.shape[0], 1))
    # plt.plot(y_true, label="true values")
    # plt.legend(loc='upper right')
    # plt.show()

    # Make multiple-step predictions
    x_input = x_test[0]
    predictions = lstm_model.predict_multiple_steps(x_input, y_test.shape[0])

    plt.plot(data_scaled, label="true values")
    plt.plot(range(data_scaled.shape[0]-y_test.shape[0], data_scaled.shape[0]), predictions, label="predictions")
    plt.axvline(x=594, color='g', linestyle='--')
    plt.title('STN Bedok')
    # plt.plot(predictions, label="predictions")
    # y_true = y_test[0:predictions.shape[0]]
    # y_true = np.reshape(y_true, (y_true.shape[0], 1))
    # plt.plot(y_true, label="true values")
    plt.legend(loc='upper right')
    plt.show()


    ############
    ## Sine data
    ############
    # x = np.linspace(0, 120, 420)
    # data = y = reader.generate_sin_signal(x, noisy=True)
    # train, val, test = reader.mrt_simple_lstm_data(data, config.train_batch_size, config.time_steps)
    # x_train, y_train = train[0], train[1]
    # x_val, y_val, = val[0], val[1]
    # x_test, y_test = test[0], test[1]
    # lstm_model.fit(x_train, y_train, x_val, y_val)
    #
    # # predictions = lstm_model.predict(x_test)
    #
    # x_input = x_test[0, :, :]
    # predictions = lstm_model.predict_multiple_steps(x_input, y_test.shape[0])
    # plt.plot(predictions[:, 0], label="predictions")
    # plt.plot(y_test[:, 0], label="true values")
    # plt.legend(loc='upper right')
    # plt.show()
    #
    # squared_error = np.square(np.subtract(predictions[:, 0], y_test[:, 0]))
    # mean_error = np.mean(squared_error)
    # rmse = np.sqrt(mean_error)
    #
    # print(rmse)


