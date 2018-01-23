import os
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

class Simple_LSTM():
    ''' Simple LSTM Model
    '''

    def __init__(self, config):

        tf.reset_default_graph()

        self.state_size = config.state_size
        self.input_size = config.input_size
        self.train_batch_size = config.train_batch_size
        self.output_size = config.output_size
        self.time_steps = config.time_steps
        self.lr = config.lr
        self.epochs = config.num_epochs
        self._sess = tf.Session()
        self._val_loss = 0
        self.checkpoint = config.checkpoint

        self._create_placeholders()
        self._create_variables()
        self._build_lstm()
        self._define_loss()
        self._define_optimizer()
        self._define_summaries()

        self._saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        self._sess.run(init)

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint))
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
                                      initializer=tf.truncated_normal([self.state_size[-1], self.output_size]))
        self._b_out = tf.get_variable("output_bias", dtype=tf.float32,
                                      initializer=tf.truncated_normal([1, self.output_size]))
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

    def _get_lstm_cells(self):
        # cell = tf.contrib.rnn.BasicLSTMCell(self.state_size)
        cells = tf.contrib.rnn.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(size) for size in self.state_size])
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

            learning_rate = tf.train.exponential_decay(self.lr, self._global_step,
                                           100, 0.7, staircase=True)
            self._optimizer = tf.train.AdamOptimizer(learning_rate)
            self._train_step = self._optimizer.minimize(self._total_loss)

    def _define_summaries(self):
        with tf.name_scope("summaries"):
            self._train_summary = tf.summary.scalar("train_rmse", self._total_loss)
            # self._val_summary = tf.summary.scalar("validation_rmse", self._val_loss)

    def save_graph(self):
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./graphs', sess.graph)
        writer.close()

    def _increment_global_step(self):
        self._sess.run(tf.assign(self._global_step, self._global_step + 1))

    def fit(self, x_train, y_train, x_val, y_val):
        '''Train the model
        :param x_train: shape [num_batches, batch_size, time_steps, feature_len]
        :param y_train: shape [num_batches, batch_size, feature len]
        :param x_val: shape [num_val_samples, time_steps, feature_len], batch_size is always 1
        :param y_val: shape [num_val_samples, feature_len], batch_size is always 1, output time step is 1
        '''

        writer = tf.summary.FileWriter("summaries/")
        num_batches = x_train.shape[0]
        best_val_loss = 1000

        for epoch in range(self.epochs):

            self._increment_global_step()

            for i in range(num_batches):
                x = x_train[i]
                y = y_train[i]
                train_loss, train_steps, train_summ = self._sess.run(
                    [self._total_loss, self._train_step, self._train_summary],
                    feed_dict={
                        self._batchX_placeholder: x,
                        self._batchY_placeholder: y,
                        self._batch_size: self.train_batch_size
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

            val_loss = self._sess.run(self._compute_rmse(y_val, val_predictions))
            summary = tf.Summary(value=[tf.Summary.Value(tag="validation_rmse",
                                                         simple_value=val_loss)])
            writer.add_summary(summary, global_step=self._sess.run(self._global_step))
            writer.flush()

            # print the errors at the end of each epoch
            print("Step", epoch, "train loss", train_loss, ", val_loss: ", val_loss)
            # print("Step", epoch, "train loss", total_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print("Achieved better val_loss. Saving model...")

                if not os.path.exists(os.path.dirname(self.checkpoint)):
                    os.makedirs(os.path.dirname(self.checkpoint))

                self._saver.save(self._sess, self.checkpoint, global_step=self._global_step)

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

    def predict_multiple_steps(self, x, time_features, y_true=None):
        '''Make multipel-steps predictions
        :param x: shape [time_step, feature_len], only input one sample
        :param num_steps: the number of feature steps to predict
        :param y_true: shape [num_test_samples, feature_len]
        :return predictions: shape [num_test_samples, feature_len]
        '''

        rmse = None
        batchX = np.reshape(x, (1, x.shape[0], x.shape[1]))
        prediction_list = []

        num_steps = time_features.shape[0]

        for j in range(num_steps):
            predictions = self._sess.run(self._prediction_series,
                                                        feed_dict={
                                                            self._batchX_placeholder: batchX,
                                                            self._batch_size: 1
                                                        })

            result = predictions.reshape(predictions.shape[1])
            prediction_list.append(result)


            time_feature = time_features[j]
            new_sample = np.array([result, time_feature[0], time_feature[1]])
            new_sample = new_sample.reshape(1, 1, new_sample.shape[0])

            batchX = np.append(batchX, new_sample, axis=1)
            batchX = np.delete(batchX, 0, axis=1)

        prediction_list = np.array(prediction_list)
        if y_true is not None:
            rmse = self._sess.run(self._compute_rmse(y_true, prediction_list))
        return prediction_list, rmse


