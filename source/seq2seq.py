import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import reader
from simple_lstm import Simple_LSTM
from configs.configs import Seq2SeqConfig

class Seq2Seq(Simple_LSTM):

    def __init__(self, config, inference=False):
        self._inference = inference
        self._input_size = config.input_size
        self._output_size = config.output_size
        super().__init__(config)

    def _create_placeholders(self):
        self._batchX_placeholder = tf.placeholder(tf.float32, [self.input_time_steps, None, self._input_size],
                                                  name="encoder_input")
        self._batchY_placeholder = tf.placeholder(tf.float32, [self.output_time_steps, None, self._output_size],
                                                  name="decoder_target")
        self._decoder_features = tf.placeholder(tf.float32, [self.output_time_steps, None, None])
        self._keep_prob_placeholder = tf.placeholder_with_default(1.0, shape=())

    def _build_model(self):
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        self._best_val_loss = tf.Variable(100, dtype=tf.float32, trainable=False, name="best_val_loss")

        _, enc_state = self._build_encoder(self._batchX_placeholder)

        with tf.variable_scope("decoder_inputs"):
            go_signal = tf.expand_dims(tf.fill(tf.shape(self._batchY_placeholder[0]), 2.0), 0)
            # go_signal = tf.zeros_like(tf.expand_dims(self._batchY_placeholder[0], 0), dtype=tf.float32, name="GO")
            dec_inp = tf.concat([go_signal, self._batchY_placeholder[:-1]], 0)
            dec_inp = tf.concat([dec_inp, self._decoder_features], axis=2)

        if not self._inference:
            dec_out = self._build_decoder(dec_inp, enc_state, False)
        else:
            dec_out = self._build_decoder(dec_inp, enc_state, True, features=self._decoder_features)

        self._prediction_series = [tf.matmul(out, self._w_out) + self._b_out for out in dec_out]

    def _build_encoder(self, enc_inp):
        with tf.variable_scope("encoder"):
            encoder_cells = self._get_lstm_cells()
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cells, enc_inp,
                                                               time_major=True, dtype=tf.float32)
        return encoder_outputs, encoder_state


    def _build_decoder(self, dec_inp, init_state, inference=False, features=None):
        if inference and features is None:
            raise ValueError("when inference is True, features cannot be None")

        with tf.variable_scope("decoder"):
            self._w_out = tf.get_variable("output_weights", dtype=tf.float32,
                                            initializer=tf.truncated_normal([self.state_size[-1], self._output_size]))
            self._b_out = tf.get_variable("output_bias", dtype=tf.float32,
                                    initializer=tf.truncated_normal([1, self._output_size]))
            self._w_dec_inp = tf.get_variable("decoder_input_weights", dtype=tf.float32,
                                              initializer=tf.truncated_normal([self._input_size, self._input_size]))
            self._b_dec_inp = tf.get_variable("decoder_input_bias", dtype=tf.float32,
                                              initializer=tf.truncated_normal([self._input_size]))

            cell = self._get_lstm_cells()

            state = init_state
            outputs = []
            prev = None
            for i in range(self.output_time_steps):
                inp = dec_inp[i]
                if inference and prev is not None:
                    with tf.variable_scope("loop_function", reuse=True):
                        inp = self._loop_function(prev, features[i])
                else:
                    inp = tf.matmul(inp, self._w_dec_inp) + self._b_dec_inp

                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                output, state = cell(inp, state)
                outputs.append(output)
                if inference:
                    prev = output

        return outputs


    def _loop_function(self, prev, current_features):
        outputs = tf.matmul(prev, self._w_out) + self._b_out
        current_inp = tf.concat([outputs, current_features], axis=1)
        return tf.matmul(current_inp, self._w_dec_inp) + self._b_dec_inp

    def fit(self, x_train, targets_train, features_train, x_val, targets_val, features_val):

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

            self._sess.run(tf.assign(self._global_step, self._global_step + 1))

            train_loss = 1000
            for i in range(num_batches):
                x = x_train[i]
                y = targets_train[i]
                train_loss, train_steps = self._sess.run(
                    [self._total_loss, self._train_step],
                    feed_dict={
                        self._batchX_placeholder: x,
                        self._batchY_placeholder: y,
                        self._decoder_features: features_train[i],
                        self._keep_prob_placeholder: train_keep_prob
                    })

                if self.write_summary:
                    summ = self._sess.run(write_op, {loss: train_loss})
                    train_writer.add_summary(summ, global_step=self._sess.run(self._global_step))
                    train_writer.flush()

            _, val_loss = self.predict(x_val, targets_val, features_val)

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

    def predict(self, x, targets, features):
        '''Make predictions
        :param x: shape [num_samples, time_step, feature_len], batch_size is always 1
        :param y:
        '''
        prediction_array = []
        loss_array = []
        num_input_windows = x.shape[0]
        for i in range(num_input_windows):
            predictions, loss = self._sess.run([self._prediction_series, self._total_loss],
                                               feed_dict={
                                                   self._batchX_placeholder: x[i],
                                                   self._batchY_placeholder: targets[i],
                                                   self._decoder_features: features[i]
                                               })
            prediction_array.append(predictions)
            loss_array.append(loss)

        prediction_array = np.array(prediction_array)
        prediction_array = prediction_array.flatten()

        rmse = np.mean(loss_array)

        return prediction_array, rmse

if __name__ == "__main__":
    # stations = [0, 8, 27, 32, 69, 75, 100, 110, 111]
    stations = [0]

    for s in stations:
        config = Seq2SeqConfig(s)
        model = Seq2Seq(config, False)

        # Load data
        data_path = "data/count_by_hour_with_header.csv"
        data_scaled = reader.get_scaled_mrt_data(data_path, [s], datetime_features=True)
        train, val, test = reader.produce_seq2seq_data(data_scaled, config.train_batch_size, config.input_time_steps,
                                                          config.output_time_steps, time_major=True, y_has_features=True)
        x_train, y_train = train[0], train[1]
        x_val, y_val, = val[0], val[1]
        x_test, y_test = test[0], test[1]

        targets_train = y_train[:, :, :, [0]]
        features_train = y_train[:, :, :, 1:]

        targets_val = y_val[:, :, :, [0]]
        features_val = y_val[:, :, :, 1:]

        targets_test = y_test[:, :, :, [0]]
        features_test = y_test[:, :, :, 1:]


        model.fit(x_train, targets_train, features_train, x_val, targets_val, features_val)

        # model = Seq2Seq(config, True)
        # predictions, rmse = model.predict(x_test, targets_test, features_test)
        # print(rmse)
        #
        # plt.plot(data_scaled[:, 0], label="true values")
        # num_test = round(data_scaled.shape[0] * 0.2)
        # plt.plot(range(data_scaled.shape[0]-num_test, data_scaled.shape[0]-num_test+predictions.size), predictions, label="predictions")
        # plt.axvline(x=data_scaled.shape[0]-num_test, color='green', linestyle='--')
        # # plt.title("STN Admiralty")
        # plt.legend(loc='upper right')
        # plt.show()