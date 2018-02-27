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
        self.feature_len = config.feature_len
        self.train_batch_size = config.train_batch_size
        self.output_time_steps = config.output_time_steps
        self.input_time_steps = config.input_time_steps
        self.lr = config.lr
        self.epochs = config.num_epochs
        self.keep_prob = config.keep_prob
        self.lr_decay = config.lr_decay
        self._sess = tf.Session()
        self._val_loss = 0
        self.checkpoint = config.checkpoint
        self.write_summary = config.write_summary
        self.tensorboard_dir = config.tensorboard_dir

        self._create_placeholders()
        self._build_model()
        self._define_loss()
        self._define_optimizer()

        self._saver = tf.train.Saver()

        init = tf.global_variables_initializer()
        self._sess.run(init)

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.checkpoint))
        if ckpt and ckpt.model_checkpoint_path:
            print("have checkpoints")
            print(ckpt.model_checkpoint_path)
            self._saver.restore(self._sess, ckpt.model_checkpoint_path)

    def _create_placeholders(self):
        self._batchX_placeholder = tf.placeholder(tf.float32, [None, self.input_time_steps, self.feature_len],
                                                  name="input")
        self._batchY_placeholder = tf.placeholder(tf.float32, [None, self.output_time_steps],
                                                  name="label")
        self._batch_size = tf.placeholder(tf.int32, [], name="batch_size")

        self._keep_prob_placeholder = tf.placeholder_with_default(1.0, shape=())

    def _get_lstm_cells(self):
        with tf.variable_scope('lstm_cells'):
            cells = [tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(size), output_keep_prob=self._keep_prob_placeholder)
                     for size in self.state_size]
            cells = tf.contrib.rnn.MultiRNNCell(cells)
            return cells

    def _build_model(self):

        self._w_out = tf.get_variable("output_weights", dtype=tf.float32,
                                      initializer=tf.truncated_normal([self.state_size[-1], self.output_time_steps]))
        self._b_out = tf.get_variable("output_bias", dtype=tf.float32,
                                      initializer=tf.truncated_normal([1, self.output_time_steps]))
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        self._best_val_loss = tf.Variable(100, dtype=tf.float32, trainable=False, name="best_val_loss")


        with tf.variable_scope("LSTM"):

            lstm_cells = self._get_lstm_cells()

            # Zero initial state
            # init_states = lstm_cells.zero_state(None, dtype=tf.float32)

            states_series, self._current_state = tf.nn.dynamic_rnn(lstm_cells, self._batchX_placeholder,
                                                                   dtype=tf.float32)

            last_time_step = states_series[:, -1, :]

            # shape [batch_size, output_time_steps]
            self._prediction_series = tf.matmul(last_time_step, self._w_out) + self._b_out

    def _compute_rmse(self, labels, predictions):
        return tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(labels, predictions))))

    def _define_loss(self):
        with tf.variable_scope("Loss"):
            self._total_loss = self._compute_rmse(self._batchY_placeholder, self._prediction_series)

    def _define_optimizer(self):
        with tf.variable_scope("Training"):

            learning_rate = tf.train.exponential_decay(self.lr, self._global_step,
                                           50, self.lr_decay, staircase=True)
            self._optimizer = tf.train.AdamOptimizer(learning_rate)
            self._train_step = self._optimizer.minimize(self._total_loss)

    def save_graph(self):
        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./graphs', sess.graph)
        writer.close()

    def _increment_global_step(self):
        self._sess.run(tf.assign(self._global_step, self._global_step + 1))


    def fit(self, x_train, y_train, x_val, y_val):
        '''Train the model
        :param x_train: shape [num_batches, batch_size, input_time_steps, feature_len]
        :param y_train: shape [num_batches, batch_size, output_time_steps]
        :param x_val: shape [num_val_samples, input_time_steps, feature_len], batch_size is always 1
        :param y_val: shape [num_val_samples, output_time_steps], batch_size is always 1
        '''


        if self.write_summary:
            train_writer = tf.summary.FileWriter(self.tensorboard_dir + "/train")
            val_writer = tf.summary.FileWriter(self.tensorboard_dir + "/validation")
            loss = tf.get_variable("loss", shape=(), dtype=tf.float32)
            tf.summary.scalar("loss", loss)
            write_op = tf.summary.merge_all()

        num_batches = x_train.shape[0]

        if self.keep_prob:
            train_keep_prob = self.keep_prob
        else:
            train_keep_prob = 1.0

        for epoch in range(self.epochs):

            self._increment_global_step()

            train_loss = 1000
            for i in range(num_batches):
                x = x_train[i]
                y = y_train[i]
                train_loss, train_steps = self._sess.run(
                    [self._total_loss, self._train_step],
                    feed_dict={
                        self._batchX_placeholder: x,
                        self._batchY_placeholder: y,
                        self._keep_prob_placeholder: train_keep_prob
                    })

                if self.write_summary:
                    summ = self._sess.run(write_op, {loss: train_loss})
                    train_writer.add_summary(summ, global_step=self._sess.run(self._global_step))
                    train_writer.flush()


            _, val_loss = self.predict(x_val, y_val)

            if self.write_summary:
                summ = self._sess.run(write_op, {loss: val_loss})
                val_writer.add_summary(summ, global_step=self._sess.run(self._global_step))
                val_writer.flush()

            print("Step", epoch, "train_loss", train_loss, ", val_loss: ", val_loss)


            if val_loss < self._sess.run(self._best_val_loss):
                self._sess.run(self._best_val_loss.assign(val_loss))
                print("Achieved better val_loss. Saving model...")

                if not os.path.exists(os.path.dirname(self.checkpoint)):
                    os.makedirs(os.path.dirname(self.checkpoint))

                self._saver.save(self._sess, self.checkpoint, global_step=self._global_step)

        if self.write_summary:
            train_writer.close()
            val_writer.close()

    def predict(self, x, y):
        '''Make predictions
        :param x: shape [num_samples, time_step, feature_len], batch_size is always 1
        :param y:
        '''
        prediction_array =[]
        loss_array =[]
        num_input_windows = x.shape[0]
        for i in range(num_input_windows):
            predictions, loss = self._sess.run([self._prediction_series, self._total_loss],
                                         feed_dict={
                                             self._batchX_placeholder: x[[i]],
                                             self._batchY_placeholder: y[[i]]

                                         })
            prediction_array.append(predictions)
            loss_array.append(loss)

        prediction_array = np.array(prediction_array)
        prediction_array = prediction_array.flatten()

        rmse = np.mean(loss_array)

        return prediction_array, rmse


    def predict_multiple_steps(self, x, time_features, y_true=None):
        '''Make multiple-steps predictions
        :param x: shape [time_step, feature_len], only input one sample
        :param num_steps: the number of feature steps to predict
        :param y_true: shape [num_test_samples, feature_len]
        :return predictions: shape [num_test_samples, feature_len]
        '''

        rmse = None
        batchX = np.reshape(x, (1, x.shape[0], x.shape[1]))
        prediction_list = []

        num_steps = time_features.shape[0]
        batch_size = x.shape[1]

        for j in range(num_steps):
            predictions = self._sess.run(self._prediction_series,
                                                        feed_dict={
                                                            self._batchX_placeholder: batchX,
                                                            self._batch_size: batch_size
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


