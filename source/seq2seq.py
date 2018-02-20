import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import reader
from simple_lstm import Simple_LSTM

class Seq2Seq(Simple_LSTM):

    def __init__(self, config, inference=False):
        super(Seq2Seq, self).__init__(config)
        self.inference = inference

    def _create_placeholders(self):
        self._encoder_inp = tf.placeholder(tf.float32, [self.input_time_steps, None, self.input_size],
                                           name="encoder_input")
        self._decoder_target = tf.placeholder(tf.float32, [self.output_time_steps, None, self.output_size],
                                              name="decoder_target")

    def _build_model(self):
        self._global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

        _, enc_state = self._build_encoder(self._encoder_inp)

        # go_signal = tf.zeros_like(tf.expand_dims(self._decoder_target[0], 0), dtype=tf.float32, name="GO")
        # dec_inp = tf.concat([go_signal, self._decoder_target], 0)

        dec_output = self._build_decoder(dec_inp, enc_state, self.inference)

        # self._model_outputs = [tf.matmul(out, self._w_out) + self._b_out for out in dec_output]

    def _build_encoder(self, enc_inp):
        with tf.variable_scope("encoder"):
            encoder_cells = self._get_lstm_cells()
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cells, enc_inp,
                                                               time_major=True, dtype=tf.float32)
        return encoder_outputs, encoder_state


    def _build_decoder(self, dec_inp, init_state, inference=False):
        state = init_state
        outputs = []

        def _get_output_weights_and_biases():
            w_out = tf.get_variable("output_weights", dtype=tf.float32,
                                    initializer=tf.truncated_normal([self.state_size, self.output_size]))
            b_out = tf.get_variable("output_bias", dtype=tf.float32,
                                    initializer=tf.truncated_normal([1, self.output_size]))
            return w_out, b_out

        if not inference:
            with tf.variable_scope("decoder"):
                self._w_out, self._b_out = _get_output_weights_and_biases()
                decoder_cells = self._get_lstm_cells()
                for i in range(self.output_time_steps):
                    inp = dec_inp[i]
                    output, state = decoder_cells(inp, state)
                    output = tf.matmul(output, self._w_out) + self._b_out
                    outputs.append(output)

        if inference:
            with tf.variable_scope("decoder", reuse=True):
                self._w_out, self._b_out = _get_output_weights_and_biases()
                decoder_cells = self._get_lstm_cells()
                state = init_state
                prev_output = None
                for i in range(self.output_time_steps):
                    inp = dec_inp[i]
                    if i == 0:
                        batch_size = inp.shape[0]
                        go_signal = tf.zeros([batch_size, 1], dtype=tf.float32, name="GO")
                        inp = tf.concat([go_signal, inp], axis=1)
                    else:
                        inp = tf.concat([prev_output, inp], axis=1)

                    output, state = decoder_cells(inp, state)
                    output = tf.matmul(output, self._w_out) + self._b_out
                    outputs.append(output)
                    prev_output = output

        return outputs


    # def _build_decoder(self, dec_inp, init_state, inference=False):
    #     with tf.variable_scope("Decoder") as dec_scope:
    #         decoder_cells = self._get_lstm_cells()
    #         state = init_state
    #         outputs = []
    #         prev_output = None
    #         for i in range(self.output_time_steps):
    #             inp = dec_inp[i]
    #             if inference and prev_output is not None:
    #                 inp = tf.matmul(prev_output, self._w_out) + self._b_out
    #             if i > 0:
    #                 dec_scope.reuse_variables()
    #             output, state = decoder_cells(inp, state)
    #             outputs.append(output)
    #             prev_output = output
    #
    #     return outputs, state