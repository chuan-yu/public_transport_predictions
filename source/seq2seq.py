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
        self._sess.run(tf.global_variables_initializer())

        # encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        # pretrain_saver = tf.train.Saver(encoder_vars)
        #
        # encoder_ckpt = tf.train.get_checkpoint_state(os.path.dirname("checkpoints/pretrain/pretrain.ckpt"))
        # if encoder_ckpt and encoder_ckpt.model_checkpoint_path:
        #     print("restoring pretrained encoder")
        #     pretrain_saver.restore(self._sess, encoder_ckpt.model_checkpoint_path)


    def _create_placeholders(self):

        # self._x_pretrain = tf.placeholder(tf.float32, [self.input_time_steps, None, self.feature_len])
        self._y_pretrain = tf.placeholder(tf.float32, [self.input_time_steps, None, 1])

        self._batchX_placeholder = tf.placeholder(tf.float32, [self.input_time_steps, None, self._input_size],
                                                  name="encoder_input")
        self._batchY_placeholder = tf.placeholder(tf.float32, [self.output_time_steps, None, self._output_size],
                                                  name="decoder_target")
        self._decoder_features = tf.placeholder(tf.float32, [self.output_time_steps, None, None])
        self._keep_prob_placeholder = tf.placeholder_with_default(1.0, shape=())

    def _build_model(self):
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        self._best_val_loss = tf.Variable(100, dtype=tf.float32, trainable=False, name="best_val_loss")

        self._encoder_outputs, enc_state = self._build_encoder(self._batchX_placeholder)

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
            self._w_encoder_out = tf.get_variable("encoder_output_weights", dtype=tf.float32,
                                                  initializer=tf.truncated_normal(([self.state_size[-1], self._output_size])))
            self._b_encoder_out = tf.get_variable("encoder_output_biases", dtype=tf.float32,
                                                  initializer=tf.truncated_normal([1, self._output_size]))

            encoder_outputs = tf.reshape(encoder_outputs, [-1, self.state_size[-1]])
            encoder_outputs = tf.matmul(encoder_outputs, self._w_encoder_out) + self._b_encoder_out
            encoder_outputs = tf.reshape(encoder_outputs, [self.input_time_steps, -1, self._output_size])
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


    # def _define_optimizer(self):
    #     with tf.variable_scope("Training"):
    #         self._optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
    #         gvs = self._optimizer.compute_gradients(self._total_loss)
    #         capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    #         self._train_step = self._optimizer.apply_gradients(capped_gvs)
    #
    #         # params = tf.trainable_variables()
    #         # gradients = tf.gradients(self._total_loss, params)
    #         # clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5)
    #         # learning_rate = tf.train.exponential_decay(self.lr, self._global_step,
    #         #                                            50, self.lr_decay, staircase=True)
    #         # self._optimizer = tf.train.AdamOptimizer(learning_rate)
    #         # self._train_step = self._optimizer.apply_gradients(zip(clipped_gradients, params))


    def pretrain_encoder(self, x, y, lr, epochs):
        encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='encoder')
        saver = tf.train.Saver(encoder_vars)

        with tf.variable_scope("pretrain_loss"):
            pretrain_loss = self._compute_rmse(self._y_pretrain, self._encoder_outputs)

        with tf.variable_scope("pre_training"):
            pretrain_optimizer = tf.train.AdamOptimizer(lr)
            pretrain_step = pretrain_optimizer.minimize(pretrain_loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            n_batches = x.shape[0]
            for epoch in range(epochs):
                for i in range(n_batches):
                    loss, _ = sess.run([pretrain_loss, pretrain_step],
                                       feed_dict={
                                           self._batchX_placeholder: x[i],
                                           self._y_pretrain: y[i]
                                       })
                    print("train_loss: ", str(loss))

                if epoch == epochs - 1:
                    print("saving pretrained encoder")
                    saver.save(sess, "checkpoints/pretrain/pretrain.ckpt")



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

    HOLIDAYS = ['2016-03-25']

    for s in stations:
        config = Seq2SeqConfig(s)
        model = Seq2Seq(config, False)

        # Load data
        data_path = "data/count_by_hour_with_header.csv"
        data_scaled = reader.get_scaled_mrt_data(data_path, [s], datetime_features=True, holidays=HOLIDAYS)

        n = data_scaled.shape[0]
        x_pretrain, y_pretrain = reader.get_pretrain_data(data_scaled[0:round(0.6 * n)],
                                                          3, config.input_time_steps)

        # model.pretrain_encoder(x_pretrain, y_pretrain, 0.0005, 2000)
        # model = Seq2Seq(config, False)
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


        # model.fit(x_train, targets_train, features_train, x_val, targets_val, features_val)

        model = Seq2Seq(config, True)
        predictions, rmse = model.predict(x_test, targets_test, features_test)
        print(rmse)
        #
        # plt.plot(data_scaled[:, 0], label="true values")
        # num_test = round(data_scaled.shape[0] * 0.2)
        # plt.plot(range(data_scaled.shape[0]-num_test, data_scaled.shape[0]-num_test+predictions.size), predictions, label="predictions")
        # plt.axvline(x=data_scaled.shape[0]-num_test, color='green', linestyle='--')
        # # plt.title("STN Admiralty")
        # plt.legend(loc='upper right')
        # plt.show()