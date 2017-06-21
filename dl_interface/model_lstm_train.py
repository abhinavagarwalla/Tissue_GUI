import shutil
from time import time
import os

import tensorflow as tf
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import random
import glob

from dl_interface.model_config import LSTMTrainConfig
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import pickle

from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple, LayerNormBasicLSTMCell
import numpy as np
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear


slim = tf.contrib.slim

def ln(tensor, scope=None, epsilon=1e-5):
    """ Layer normalizes a 2D tensor along its second axis """
    assert (len(tensor.get_shape()) == 2)
    m, v = tf.nn.moments(tensor, [1], keep_dims=True)
    if not isinstance(scope, str):
        scope = ''
    with tf.variable_scope(scope + 'layer_norm'):
        scale = tf.get_variable('scale',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('shift',
                                shape=[tensor.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    ln_initial = (tensor - m) / tf.sqrt(v + epsilon)

    return ln_initial * scale + shift

class MultiDimensionalLSTMCell(RNNCell):
    """
    Adapted from TF's BasicLSTMCell to use Layer Normalization.
    Note that state_is_tuple is always True.
    """

    def __init__(self, num_units, forget_bias=0.0, activation=tf.nn.tanh):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM).
        @param: inputs (batch,n)
        @param state: the states and hidden unit of the two cells
        """
        with tf.variable_scope(scope or type(self).__name__):
            c1, c2, h1, h2 = state

            # change bias argument to False since LN will add bias via shift
            concat = _linear([inputs, h1, h2], 5 * self._num_units, False)

            i, j, f1, f2, o = tf.split(value=concat, num_or_size_splits=5, axis=1)

            # add layer normalization to each gate
            i = ln(i, scope='i/')
            j = ln(j, scope='j/')
            f1 = ln(f1, scope='f1/')
            f2 = ln(f2, scope='f2/')
            o = ln(o, scope='o/')

            new_c = (c1 * tf.nn.sigmoid(f1 + self._forget_bias) +
                     c2 * tf.nn.sigmoid(f2 + self._forget_bias) + tf.nn.sigmoid(i) *
                     self._activation(j))

            # add layer_normalization in calculation of new hidden state
            new_h = self._activation(ln(new_c, scope='new_h/')) * tf.nn.sigmoid(o)
            new_state = LSTMStateTuple(new_c, new_h)

            return new_h, new_state

class DataIter():
    def __init__(self):
        self.wsi_list = glob.glob(LSTMTrainConfig.DATA_IMAGES_PATH + os.sep + '*')
        self.images_list = []
        for i in self.wsi_list:
            self.images_list.extend(glob.glob(i + os.sep + '*'))
        random.shuffle(self.images_list)
        self.num_samples = len(self.images_list)
        self.iter = 0

    def next_batch(self):
        x = []
        y = []
        for i in range(LSTMTrainConfig.batch_size):
            feat = pickle.load(open(self.images_list[self.iter], 'rb'))
            wsi_name = self.images_list[self.iter].split(os.sep)[-1]
            label = np.load(LSTMTrainConfig.DATA_LABELS_PATH + os.sep + wsi_name.split('_(')[0] +
                            os.sep + wsi_name.replace('features.pkl', 'label.npy'))
            x.append(feat['fc6'].reshape(LSTMTrainConfig.PATCH_SIZE, LSTMTrainConfig.PATCH_SIZE, LSTMTrainConfig.CHANNELS))
            y.append(label)
            self.iter += 1
            if self.iter == self.num_samples:
                break
        return x, y

class LSTMTrain(QObject):
    finished = pyqtSignal()
    epoch = pyqtSignal(int)

    def initialize(self):
        self.t0 = time()
        self.dataloader = DataIter()

    def multi_dimensional_rnn_while_loop(self, rnn_size, input_data, sh, dims=None, scope_n="layer1"):
        """Implements naive multi dimension recurrent neural networks

        @param rnn_size: the hidden units
        @param input_data: the data to process of shape [batch,h,w,channels]
        @param sh: [height,width] of the windows
        @param dims: dimensions to reverse the input data,eg.
            dims=[False,True,True,False] => true means reverse dimension
        @param scope_n : the scope

        returns [batch,h/sh[0],w/sh[1],channels*sh[0]*sh[1]] the output of the lstm
        """
        with tf.variable_scope("MultiDimensionalLSTMCell-" + scope_n):
            cell = MultiDimensionalLSTMCell(rnn_size)

            shape = input_data.get_shape().as_list()

            if shape[1] % sh[0] != 0:
                offset = tf.zeros([shape[0], sh[0] - (shape[1] % sh[0]), shape[2], shape[3]])
                input_data = tf.concat(1, [input_data, offset])
                shape = input_data.get_shape().as_list()
            if shape[2] % sh[1] != 0:
                offset = tf.zeros([shape[0], shape[1], sh[1] - (shape[2] % sh[1]), shape[3]])
                input_data = tf.concat(2, [input_data, offset])
                shape = input_data.get_shape().as_list()

            h, w = int(shape[1] / sh[0]), int(shape[2] / sh[1])
            features = sh[1] * sh[0] * shape[3]
            batch_size = shape[0]

            x = tf.reshape(input_data, [batch_size, h, w, features])
            if dims is not None:
                assert dims[0] is False and dims[3] is False
                for i in range(len(dims)):
                    if dims[i]:
                        x = tf.reverse(x, [i])
            x = tf.transpose(x, [1, 2, 0, 3])
            x = tf.reshape(x, [-1, features])
            x = tf.split(axis=0, num_or_size_splits=h * w, value=x)

            sequence_length = tf.ones(shape=(batch_size,), dtype=tf.int32) * shape[0]
            inputs_ta = tf.TensorArray(dtype=tf.float32, size=h * w, name='input_ta')
            inputs_ta = inputs_ta.unstack(x)
            states_ta = tf.TensorArray(dtype=tf.float32, size=h * w + 1, name='state_ta', clear_after_read=False)
            outputs_ta = tf.TensorArray(dtype=tf.float32, size=h * w, name='output_ta')

            # initial cell and hidden states
            states_ta = states_ta.write(h * w, LSTMStateTuple(tf.zeros([batch_size, rnn_size], tf.float32),
                                                              tf.zeros([batch_size, rnn_size], tf.float32)))

            def get_up(t_, w_):
                return t_ - tf.constant(w_)

            def get_last(t_, w_):
                return t_ - tf.constant(1)

            # def get_up_last(t_, w_):
            #     return t_ - tf.constant(w_) - tf.constant(1)

            time = tf.constant(0)
            zero = tf.constant(0)

            def body(time_, outputs_ta_, states_ta_):
                state_up = tf.cond(tf.less_equal(tf.constant(w), time_),
                                   lambda: states_ta_.read(get_up(time_, w)),
                                   lambda: states_ta_.read(h * w))
                state_last = tf.cond(tf.less(zero, tf.mod(time_, tf.constant(w))),
                                     lambda: states_ta_.read(get_last(time_, w)),
                                     lambda: states_ta_.read(h * w))

                # state_up_last = tf.cond(tf.less(zero, tf.mod(time_, tf.constant(w))),
                #                         tf.cond(tf.less_equal(tf.constant(w), time_),
                #                                 lambda: states_ta_.read(get_up_last(time_, w)),
                #                                 lambda: states_ta_.read(h * w)), lambda: states_ta_.read(h * w))

                current_state = state_up[0], state_last[0], state_up[1], state_last[1]
                out, state = cell(inputs_ta.read(time_), current_state)
                outputs_ta_ = outputs_ta_.write(time_, out)
                states_ta_ = states_ta_.write(time_, state)
                return time_ + 1, outputs_ta_, states_ta_

            def condition(time_, outputs_ta_, states_ta_):
                return tf.less(time_, tf.constant(h * w))

            result, outputs_ta, states_ta = tf.while_loop(condition, body, [time, outputs_ta, states_ta],
                                                          parallel_iterations=1)

            outputs = outputs_ta.stack()
            states = states_ta.stack()

            y = tf.reshape(outputs, [h, w, batch_size, rnn_size])
            y = tf.transpose(y, [2, 0, 1, 3])
            if dims is not None:
                for i in range(len(dims)):
                    if dims[i]:
                        y = tf.reverse(y, [i])

            return y, states

    @pyqtSlot()
    def train(self):
        # Saver and initialisation
        print("starting training")
        self.initialize()
        # saver = tf.train.Saver()
        self.epoch.emit(0)
        if not os.path.exists(LSTMTrainConfig.log_dir):
            os.mkdir(LSTMTrainConfig.log_dir)

        # ======================= TRAINING PROCESS =========================
        # Now we start to construct the graph and build our model
        # Create the model inference
        tf.logging.set_verbosity(tf.logging.INFO)
        logging.info("starting with tf")
        images = tf.placeholder(tf.float32, [LSTMTrainConfig.batch_size, LSTMTrainConfig.PATCH_SIZE,
                                             LSTMTrainConfig.PATCH_SIZE, LSTMTrainConfig.CHANNELS])
        labels = tf.placeholder(tf.float32, [LSTMTrainConfig.batch_size, LSTMTrainConfig.PATCH_SIZE,
                                             LSTMTrainConfig.PATCH_SIZE, 1])
        rnn_out_1, _ = self.multi_dimensional_rnn_while_loop(rnn_size=LSTMTrainConfig.HIDDEN_SIZE, input_data=images,
                                                           sh=[1, 1], scope_n="lstm_1")
        rnn_out_2, _ = self.multi_dimensional_rnn_while_loop(rnn_size=LSTMTrainConfig.HIDDEN_SIZE, input_data=images,
                                                             sh=[1, 1], dims=[False, True, False, False], scope_n="lstm_2")
        rnn_out_3, _ = self.multi_dimensional_rnn_while_loop(rnn_size=LSTMTrainConfig.HIDDEN_SIZE, input_data=images,
                                                             sh=[1, 1], dims=[False, True, True, False], scope_n="lstm_3")
        rnn_out_4, _ = self.multi_dimensional_rnn_while_loop(rnn_size=LSTMTrainConfig.HIDDEN_SIZE, input_data=images,
                                                             sh=[1, 1], dims=[False, False, True, False], scope_n="lstm_4")

        logging.info("Four LSTMs formed")
        # model_out = slim.fully_connected(inputs=rnn_out,
        #                                  num_outputs=1,
        #                                  activation_fn=None)
        model_out_1 = slim.conv2d(inputs=rnn_out_1, num_outputs=1, kernel_size=[3,3])
        model_out_2 = slim.conv2d(inputs=rnn_out_2, num_outputs=1, kernel_size=[3, 3])
        model_out_3 = slim.conv2d(inputs=rnn_out_3, num_outputs=1, kernel_size=[3, 3])
        model_out_4 = slim.conv2d(inputs=rnn_out_4, num_outputs=1, kernel_size=[3, 3])

        logging.info("Convolution done, reshaping")
        # model_out_2 = tf.reverse(model_out_2, axis=[False, True, False, False])
        # model_out_3 = tf.reverse(model_out_3, axis=[False, True, True, False])
        # model_out_4 = tf.reverse(model_out_4, axis=[False, False, True, False])

        logging.info("Doing Scalar")
        model_out = tf.scalar_mul(tf.constant(0.25),tf.add_n([model_out_1, model_out_2, model_out_3, model_out_4]))
        loss = 1e4 * tf.reduce_mean(tf.abs(tf.subtract(labels, model_out)))
        grad_update = tf.train.AdamOptimizer(LSTMTrainConfig.initial_learning_rate).minimize(loss)

        # Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        saver = tf.train.Saver(max_to_keep=None)

        logging.info("now starting session")
        # Run the managed session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            logging.info("initialiser run")
            for step in range(int(1000 * LSTMTrainConfig.num_epochs)):
                batch_x, batch_y = self.dataloader.next_batch()

                # Log the summaries every 10 step.
                if step % 100 == 0:
                    loss_value, _ = sess.run([loss, grad_update], feed_dict={images: batch_x, labels: batch_y})
                else:
                    loss_value, _ = sess.run([loss, grad_update], feed_dict={images: batch_x, labels: batch_y})

                logging.info("At step %d/%d, loss= %.4f", step, int(1000 * LSTMTrainConfig.num_epochs), loss_value)
                if step % 500==0:
                    saver.save(sess, LSTMTrainConfig.log_dir + os.sep + "lstm_model", global_step=step)
            saver.save(sess, LSTMTrainConfig.log_dir + os.sep + "lstm_model",global_step=step)
            self.finished.emit()

    @pyqtSlot()
    def stop_call(self):
        print("Stopping Training..")
        self.epoch.emit(0)
        self.finished.emit()