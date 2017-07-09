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
from nets import nets_factory

slim = tf.contrib.slim

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
        model_out = nets_factory.get_network_fn('Stacked-2D-LSTM', images, num_classes=LSTMValidConfig.NUM_CLASSES,
                                                is_training=False)

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
        model_tumor_preds = tf.reshape(tf.slice(tf.nn.softmax(model_out_flat), [0, 1], [si, 1]), [si])
        cnn_tumor_preds = tf.reshape(tf.slice(tf.nn.softmax(cnn_preds_flat), [0, 1], [si, 1]), [si])

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
        sv = tf.train.Supervisor(logdir=LSTMValidConfig.log_dir, summary_op=None, init_fn=restore_fn, saver=saver_all)

        logging.info("now starting session")
        # Run the managed session
        with sv.managed_session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.90))) as sess:
            logging.info("initialiser run")
            for step in range(int(num_steps_per_epoch)):
                batch_x, batch_y, cnn_y, cnn_logits = self.dataloader.next_batch()

                # Log the summaries every 10 step.
                loss_value, loss_cnn_value, model_tumor_preds_value, summaries,\
                global_step_count, _1, _2, acc_value, acc_value_cnn = sess.run([loss, loss_cnn, model_tumor_preds, my_summary_op,
                            sv.global_step, metrics_op, metrics_op_cnn, accuracy_batch, accuracy_batch_cnn],
                            feed_dict={images: batch_x, labels: batch_y, cnn_preds: cnn_logits})
                sv.summary_computed(sess, summaries, global_step=step)

                logging.info("At step %d/%d, loss= %.4f, accuracy=%.2f; cnn_only_loss= %.4f, cnn_only_accuracy=%.2f",
                             step, int(num_steps_per_epoch * LSTMValidConfig.num_epochs),
                             loss_value, 100*acc_value, loss_cnn_value, 100*acc_value_cnn)
                self.dataloader.save_predictions(model_tumor_preds_value)
            logging.info('Finished validation! Saving model to disk now.')
            self.finished.emit()