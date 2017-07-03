# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from time import time
import os
import random
import glob

import tensorflow as tf
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from dl_interface.model_config import LSTMValidConfig
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
        self.wsi_list = glob.glob(LSTMValidConfig.DATA_IMAGES_PATH + os.sep + '*')
        self.images_list = []
        for i in self.wsi_list:
            self.images_list.extend(glob.glob(i + os.sep + '*'))
        random.shuffle(self.images_list)
        self.num_samples = len(self.images_list)
        self.iter = 0

    def next_batch(self):
        x = []
        y = []
        cnn_logits = []
        cnn_y = []
        self.names = []
        for i in range(LSTMValidConfig.batch_size):
            feat = pickle.load(open(self.images_list[self.iter], 'rb'))
            wsi_name = self.images_list[self.iter].split(os.sep)[-1]
            label = np.load(LSTMValidConfig.DATA_LABELS_PATH + os.sep + wsi_name.split('_(')[0] +
                            os.sep + wsi_name.replace('features.pkl', 'label.npy'))
            x.append(feat['fc6'].reshape(LSTMValidConfig.PATCH_SIZE, LSTMValidConfig.PATCH_SIZE, LSTMValidConfig.CHANNELS))
            y.append(label)
            cnn_y.append(feat['predictions'].reshape(LSTMValidConfig.PATCH_SIZE, LSTMValidConfig.PATCH_SIZE, 1))
            cnn_logits.append(feat['fc8'].reshape(LSTMValidConfig.PATCH_SIZE, LSTMValidConfig.PATCH_SIZE, LSTMValidConfig.NUM_CLASSES))
            self.names.append(self.images_list[self.iter].replace('_features.pkl', '_preds.npy').split(os.sep)[-1])
            self.iter += 1
            if self.iter == self.num_samples:
                self.iter = 0
        return x, y, cnn_y, cnn_logits

    def save_predictions(self, probs):
        for i in range(len(self.names)):
            start = i*LSTMValidConfig.PATCH_SIZE*LSTMValidConfig.PATCH_SIZE
            end = (i+1)*LSTMValidConfig.PATCH_SIZE*LSTMValidConfig.PATCH_SIZE
            np.save(LSTMValidConfig.log_dir + os.sep + 'predictions' + os.sep + self.names[i], probs[start:end,])

class StackedLSTMValidation(QObject):
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
    def valid(self):
        # Saver and initialisation
        print("starting Stacked LSTM Model Validation")
        self.initialize()
        self.epoch.emit(0)
        if not os.path.exists(LSTMValidConfig.log_dir):
            os.mkdir(LSTMValidConfig.log_dir)

        # ======================= TRAINING PROCESS =========================
        # Now we start to construct the graph and build our model
        # Create the model inference
        tf.logging.set_verbosity(tf.logging.INFO)
        logging.info("starting with tf")

        # Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = self.dataloader.num_samples / LSTMValidConfig.batch_size
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed


        images = tf.placeholder(tf.float32, [LSTMValidConfig.batch_size, LSTMValidConfig.PATCH_SIZE,
                                             LSTMValidConfig.PATCH_SIZE, LSTMValidConfig.CHANNELS])
        labels = tf.placeholder(tf.float32, [LSTMValidConfig.batch_size, LSTMValidConfig.PATCH_SIZE,
                                             LSTMValidConfig.PATCH_SIZE, 1])#
        cnn_preds = tf.placeholder(tf.float32, [LSTMValidConfig.batch_size, LSTMValidConfig.PATCH_SIZE,
                                                LSTMValidConfig.PATCH_SIZE, LSTMValidConfig.NUM_CLASSES])
        rnn_out_1, _ = self.multi_dimensional_rnn_while_loop(rnn_size=LSTMValidConfig.HIDDEN_SIZE, input_data=images,
                                                             sh=[1, 1], scope_n="lstm_1")
        rnn_out_2, _ = self.multi_dimensional_rnn_while_loop(rnn_size=LSTMValidConfig.HIDDEN_SIZE, input_data=images,
                                                             sh=[1, 1], dims=[False, True, False, False],
                                                             scope_n="lstm_2")
        rnn_out_3, _ = self.multi_dimensional_rnn_while_loop(rnn_size=LSTMValidConfig.HIDDEN_SIZE, input_data=images,
                                                             sh=[1, 1], dims=[False, True, True, False],
                                                             scope_n="lstm_3")
        rnn_out_4, _ = self.multi_dimensional_rnn_while_loop(rnn_size=LSTMValidConfig.HIDDEN_SIZE, input_data=images,
                                                             sh=[1, 1], dims=[False, False, True, False],
                                                             scope_n="lstm_4")

        logging.info("Four LSTMs formed")
        model_out_1 = slim.conv2d(inputs=rnn_out_1, num_outputs=LSTMValidConfig.HIDDEN_SIZE,
                                  kernel_size=[3, 3])  # , activation_fn=None)
        model_out_2 = slim.conv2d(inputs=rnn_out_2, num_outputs=LSTMValidConfig.HIDDEN_SIZE,
                                  kernel_size=[3, 3])  # , activation_fn=None)
        model_out_3 = slim.conv2d(inputs=rnn_out_3, num_outputs=LSTMValidConfig.HIDDEN_SIZE,
                                  kernel_size=[3, 3])  # , activation_fn=None)
        model_out_4 = slim.conv2d(inputs=rnn_out_4, num_outputs=LSTMValidConfig.HIDDEN_SIZE,
                                  kernel_size=[3, 3])  # , activation_fn=None)
        stack_out = tf.scalar_mul(tf.constant(0.25), tf.add_n([model_out_1, model_out_2, model_out_3, model_out_4]))

        s2_rnn_out_1, _ = self.multi_dimensional_rnn_while_loop(rnn_size=LSTMValidConfig.HIDDEN_SIZE,
                                                                input_data=stack_out,
                                                                sh=[1, 1], scope_n="s2_lstm_1")
        s2_rnn_out_2, _ = self.multi_dimensional_rnn_while_loop(rnn_size=LSTMValidConfig.HIDDEN_SIZE,
                                                                input_data=stack_out,
                                                                sh=[1, 1], dims=[False, True, False, False],
                                                                scope_n="s2_lstm_2")
        s2_rnn_out_3, _ = self.multi_dimensional_rnn_while_loop(rnn_size=LSTMValidConfig.HIDDEN_SIZE,
                                                                input_data=stack_out,
                                                                sh=[1, 1], dims=[False, True, True, False],
                                                                scope_n="s2_lstm_3")
        s2_rnn_out_4, _ = self.multi_dimensional_rnn_while_loop(rnn_size=LSTMValidConfig.HIDDEN_SIZE,
                                                                input_data=stack_out,
                                                                sh=[1, 1], dims=[False, False, True, False],
                                                                scope_n="s2_lstm_4")

        logging.info("Stacked LSTMs formed")
        s2_model_out_1 = slim.conv2d(inputs=s2_rnn_out_1, num_outputs=LSTMValidConfig.NUM_CLASSES, kernel_size=[3, 3],
                                     activation_fn=None)
        s2_model_out_2 = slim.conv2d(inputs=s2_rnn_out_2, num_outputs=LSTMValidConfig.NUM_CLASSES, kernel_size=[3, 3],
                                     activation_fn=None)
        s2_model_out_3 = slim.conv2d(inputs=s2_rnn_out_3, num_outputs=LSTMValidConfig.NUM_CLASSES, kernel_size=[3, 3],
                                     activation_fn=None)
        s2_model_out_4 = slim.conv2d(inputs=s2_rnn_out_4, num_outputs=LSTMValidConfig.NUM_CLASSES, kernel_size=[3, 3],
                                     activation_fn=None)
        model_out = tf.scalar_mul(tf.constant(0.25),
                                  tf.add_n([s2_model_out_1, s2_model_out_2, s2_model_out_3, s2_model_out_4]))

        model_out_flat = tf.reshape(model_out, shape=(-1, LSTMValidConfig.NUM_CLASSES))

        non_tumor_label = tf.subtract(tf.ones((LSTMValidConfig.PATCH_SIZE, LSTMValidConfig.PATCH_SIZE, 1)), labels)
        combined_label = tf.concat([non_tumor_label, labels], axis=3)
        labels_flat = tf.reshape(combined_label, shape=(-1,LSTMValidConfig.NUM_CLASSES))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_out_flat,
                                                                  labels=labels_flat))

        cnn_preds_flat = tf.reshape(cnn_preds, shape=(-1, LSTMValidConfig.NUM_CLASSES))
        loss_cnn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cnn_preds_flat,
                                                                  labels=labels_flat))

        model_out_class = tf.argmax(model_out_flat, axis=1)
        cnn_preds_class = tf.argmax(cnn_preds_flat, axis=1)
        labels_class = tf.argmax(labels_flat, axis=1)

        si = int(model_out_flat.get_shape()[0])
        model_tumor_preds = tf.reshape(tf.slice(model_out_flat, [0, 1], [si, 1]), [si])
        cnn_tumor_preds = tf.reshape(tf.slice(cnn_preds_flat, [0, 1], [si, 1]), [si])

        accuracy_streaming, accuracy_streaming_update = tf.contrib.metrics.streaming_accuracy(model_out_class, labels_class)
        precision_streaming, precision_streaming_update = tf.contrib.metrics.streaming_precision(model_out_class, labels_class)
        recall_streaming, recall_streaming_update = tf.contrib.metrics.streaming_recall(model_out_class, labels_class)
        precision_thresh, precision_thresh_update = tf.contrib.metrics.streaming_precision_at_thresholds(model_tumor_preds,
                                                               labels_class,
                                                                np.arange(0.0,1.0,0.05).astype(np.float32))
        recall_thresh, recall_thresh_update = tf.contrib.metrics.streaming_recall_at_thresholds(model_tumor_preds,
                                                                                               labels_class,
                                                                                               np.arange(0.0, 1.0,
                                                                                                         0.05).astype(
                                                                                                   np.float32))
        accuracy_batch, accuracy_batch_update = tf.metrics.accuracy(labels_class, model_out_class)
        precision_batch, precision_batch_update = tf.metrics.precision(labels_class, model_out_class)
        recall_batch, recall_batch_update = tf.metrics.recall(labels_class, model_out_class)
        metrics_op = tf.group(recall_streaming_update, precision_streaming_update, accuracy_streaming_update,
                              recall_batch_update, precision_batch_update, accuracy_batch_update,
                              precision_thresh_update, recall_thresh_update)

        accuracy_streaming_cnn, accuracy_streaming_cnn_update = tf.contrib.metrics.streaming_accuracy(cnn_preds_class,
                                                                                              labels_class)
        precision_streaming_cnn, precision_streaming_cnn_update = tf.contrib.metrics.streaming_precision(cnn_preds_class,
                                                                                                 labels_class)
        recall_streaming_cnn, recall_streaming_cnn_update = tf.contrib.metrics.streaming_recall(cnn_preds_class, labels_class)
        precision_thresh_cnn, precision_thresh_cnn_update = tf.contrib.metrics.streaming_precision_at_thresholds(cnn_tumor_preds,
                                                                                                                labels_class,
                                                                                                                np.arange(
                                                                                                                    0.0,
                                                                                                                    1.0,
                                                                                                                    0.05).astype(
                                                                                                                    np.float32))
        recall_thresh_cnn, recall_thresh_cnn_update = tf.contrib.metrics.streaming_recall_at_thresholds(cnn_tumor_preds,
                                                                                                       labels_class,
                                                                                                       np.arange(0.0,
                                                                                                                 1.0,
                                                                                                                 0.05).astype(
                                                                                                           np.float32))
        accuracy_batch_cnn, accuracy_batch_cnn_update = tf.metrics.accuracy(labels_class, cnn_preds_class)
        precision_batch_cnn, precision_batch_cnn_update = tf.metrics.precision(labels_class, cnn_preds_class)
        recall_batch_cnn, recall_batch_cnn_update = tf.metrics.recall(labels_class, cnn_preds_class)
        metrics_op_cnn = tf.group(recall_streaming_cnn_update, precision_streaming_cnn_update, accuracy_streaming_cnn_update,
                              recall_batch_cnn_update, precision_batch_cnn_update, accuracy_batch_cnn_update,
                                  precision_thresh_cnn_update, recall_thresh_cnn_update)
        # loss = 1e4 * tf.reduce_mean(tf.abs(tf.subtract(labels, model_out)))
        # loss_cnn = 1e4 * tf.reduce_mean(tf.abs(tf.subtract(labels, cnn_preds)))

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        # Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Absolute_L1_Loss', loss)
        tf.summary.scalar('losses/accuracy_streaming', accuracy_streaming)
        tf.summary.scalar('losses/precision_streaming', precision_streaming)
        tf.summary.scalar('losses/recall_streaming', recall_streaming)
        tf.summary.scalar('losses/accuracy_batch', accuracy_batch)
        tf.summary.scalar('losses/precision_batch', precision_batch)
        tf.summary.scalar('losses/recall_batch', recall_batch)
        [tf.summary.scalar('precision/precision_' + str(i / 20), precision_thresh[i]) for i in range(20)]
        [tf.summary.scalar('recall/recall_' + str(i / 20), recall_thresh[i]) for i in range(20)]
        [tf.summary.scalar('f1/f1_' + str(i / 20), 2*precision_thresh[i]*recall_thresh[i]/(precision_thresh[i]+recall_thresh[i])) for i in range(20)]
        # tf.summary.scalar('thresh/precision_thresh', precision_thresh)
        # tf.summary.scalar('thresh/recall_thresh', recall_thresh)

        tf.summary.scalar('losses/Absolute_L1_Loss_CNN', loss_cnn)
        tf.summary.scalar('losses/accuracy_streaming_cnn', accuracy_streaming_cnn)
        tf.summary.scalar('losses/precision_streaming_cnn', precision_streaming_cnn)
        tf.summary.scalar('losses/recall_streaming_cnn', recall_streaming_cnn)
        tf.summary.scalar('losses/accuracy_batch_cnn', accuracy_batch_cnn)
        tf.summary.scalar('losses/precision_batch_cnn', precision_batch_cnn)
        tf.summary.scalar('losses/recall_batch_cnn', recall_batch_cnn)
        [tf.summary.scalar('precision_cnn/precision_cnn_' + str(i / 20), precision_thresh_cnn[i]) for i in range(20)]
        [tf.summary.scalar('recall_cnn/recall_cnn_' + str(i / 20), recall_thresh_cnn[i]) for i in range(20)]
        [tf.summary.scalar('f1/f1_' + str(i / 20),
                2 * precision_thresh_cnn[i] * recall_thresh_cnn[i] / (precision_thresh_cnn[i] + recall_thresh_cnn[i])) for i in range(20)]
        # tf.summary.histogram('thresh/precision_thresh_cnn', precision_thresh_cnn)
        # tf.summary.histogram('thresh/recall_thresh_cnn', recall_thresh_cnn)

        my_summary_op = tf.summary.merge_all()

        # Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        saver_all = tf.train.Saver(max_to_keep=None)

        def restore_fn(sess):
            return saver_all.restore(sess, LSTMValidConfig.checkpoint_file)

        # Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=LSTMValidConfig.log_dir, summary_op=None, init_fn=restore_fn)

        logging.info("now starting session")
        # Run the managed session
        with sv.managed_session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20))) as sess:
            logging.info("initialiser run")
            for step in range(int(num_steps_per_epoch)):
                batch_x, batch_y, cnn_y, cnn_logits = self.dataloader.next_batch()

                # Log the summaries every 10 step.
                loss_value, loss_cnn_value, model_out_flat_value, summaries,\
                global_step_count, _1, _2, acc_value, acc_value_cnn = sess.run([loss, loss_cnn, model_out_flat, my_summary_op,
                            sv.global_step, metrics_op, metrics_op_cnn, accuracy_batch, accuracy_batch_cnn],
                            feed_dict={images: batch_x, labels: batch_y, cnn_preds: cnn_logits})
                sv.summary_computed(sess, summaries, global_step=step)

                logging.info("At step %d/%d, loss= %.4f, accuracy=%.2f; cnn_only_loss= %.4f, cnn_only_accuracy=%.2f",
                             step, int(num_steps_per_epoch * LSTMValidConfig.num_epochs),
                             loss_value, 100*acc_value, loss_cnn_value, 100*acc_value_cnn)
                self.dataloader.save_predictions(model_out_flat_value)
            logging.info('Finished validation! Saving model to disk now.')
            self.finished.emit()