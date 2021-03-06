# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides class for training a CNN"""

from time import time
import os

import tensorflow as tf
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from nets import nets_factory
from dataio.cnn_batch_iter import CNNDataIter

from dl_interface.model_config import CNN2TrainConfig
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging

slim = tf.contrib.slim

class CNN2Train(QObject):
    finished = pyqtSignal()
    epoch = pyqtSignal(int)

    def initialize(self):
        self.t0 = time()
        self.dataloader = CNNDataIter()

    @pyqtSlot()
    def train(self):
        """Start training of CNN"""
        # Saver and initialisation
        print("starting CNN training")
        self.initialize()
        # saver = tf.train.Saver()
        self.epoch.emit(0)
        if not os.path.exists(CNN2TrainConfig.log_dir):
            os.mkdir(CNN2TrainConfig.log_dir)

        # ======================= TRAINING PROCESS =========================
        # Now we start to construct the graph and build our model
        # Create the model inference
        tf.logging.set_verbosity(tf.logging.INFO)
        logging.info("starting with tf")

        # Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = self.dataloader.num_samples*CNN2TrainConfig.PATCH_SIZE*CNN2TrainConfig.PATCH_SIZE / CNN2TrainConfig.batch_size
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        decay_steps = int(CNN2TrainConfig.num_epochs_before_decay * num_steps_per_epoch)


        images = tf.placeholder(tf.float32, [CNN2TrainConfig.batch_size, CNN2TrainConfig.IMAGE_SIZE,
                                             CNN2TrainConfig.IMAGE_SIZE, 3])
        labels = tf.placeholder(tf.float32, [CNN2TrainConfig.batch_size, CNN2TrainConfig.NUM_CLASSES])
        logging.info("Doing Scalar")

        cnn_preds, end_points = nets_factory.get_network_fn(name='alexnet', images=images,
                                                         num_classes=CNN2TrainConfig.NUM_CLASSES, is_training=True)

        loss_cnn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cnn_preds,
                                                                  labels=labels))

        cnn_preds_class = tf.argmax(cnn_preds, axis=1)
        labels_class = tf.argmax(labels, axis=1)

        accuracy_streaming_cnn, accuracy_streaming_cnn_update = tf.contrib.metrics.streaming_accuracy(cnn_preds_class,
                                                                                              labels_class)
        precision_streaming_cnn, precision_streaming_cnn_update = tf.contrib.metrics.streaming_precision(cnn_preds_class,
                                                                                                 labels_class)
        recall_streaming_cnn, recall_streaming_cnn_update = tf.contrib.metrics.streaming_recall(cnn_preds_class, labels_class)
        # accuracy_batch_cnn, accuracy_batch_cnn_update = tf.metrics.accuracy(labels_class, cnn_preds_class)
        # precision_batch_cnn, precision_batch_cnn_update = tf.metrics.precision(labels_class, cnn_preds_class)
        # recall_batch_cnn, recall_batch_cnn_update = tf.metrics.recall(labels_class, cnn_preds_class)
        tp_streaming_cnn, tp_streaming_cnn_update = tf.contrib.metrics.streaming_true_positives(cnn_preds_class,
                                                                                                labels_class)
        tn_streaming_cnn, tn_streaming_cnn_update = tf.contrib.metrics.streaming_true_negatives(cnn_preds_class,
                                                                                                labels_class)
        fp_streaming_cnn, fp_streaming_cnn_update = tf.contrib.metrics.streaming_false_positives(cnn_preds_class,
                                                                                                labels_class)
        fn_streaming_cnn, fn_streaming_cnn_update = tf.contrib.metrics.streaming_false_negatives(cnn_preds_class,
                                                                                                labels_class)
        metrics_op_cnn = tf.group(recall_streaming_cnn_update, precision_streaming_cnn_update, accuracy_streaming_cnn_update,
                                  tp_streaming_cnn_update, tn_streaming_cnn_update, fp_streaming_cnn_update, fn_streaming_cnn_update)

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        # Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate=CNN2TrainConfig.initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=CNN2TrainConfig.learning_rate_decay_factor,
            staircase=True)

        # Now we can define the optimizer that takes on the learning rate
        grad_update = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_cnn)

        # Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Absolute_L1_Loss_CNN', loss_cnn)
        tf.summary.scalar('accuracy_streaming_cnn', accuracy_streaming_cnn)
        tf.summary.scalar('precision_streaming_cnn', precision_streaming_cnn)
        tf.summary.scalar('recall_streaming_cnn', recall_streaming_cnn)
        tf.summary.scalar('tp_streaming_cnn', tp_streaming_cnn)
        tf.summary.scalar('tn_streaming_cnn', tn_streaming_cnn)
        tf.summary.scalar('fp_streaming_cnn', fp_streaming_cnn)
        tf.summary.scalar('fn_streaming_cnn', fn_streaming_cnn)

        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()

        # Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        saver_all = tf.train.Saver(max_to_keep=None)

        def restore_fn(sess):
            return saver_all.restore(sess, CNN2TrainConfig.checkpoint_file)

        # Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=CNN2TrainConfig.log_dir, summary_op=None, init_fn=None, saver=saver_all)  # restore_fn)

        logging.info("now starting session")
        # Run the managed session
        with sv.managed_session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20))) as sess:
            logging.info("initialiser run")
            for step in range(int(num_steps_per_epoch * CNN2TrainConfig.num_epochs)):
                batch_x, batch_y = self.dataloader.next_batch()

                # Log the summaries every 10 step.
                if step % 100 == 0:
                    loss_cnn_value, _, summaries,\
                    global_step_count, _1, acc_value_cnn = sess.run([loss_cnn, grad_update, my_summary_op,
                                sv.global_step,  metrics_op_cnn, accuracy_streaming_cnn],
                                feed_dict={images: batch_x, labels: batch_y})
                    sv.summary_computed(sess, summaries, global_step=step)
                else:
                    loss_cnn_value, _, \
                    global_step_count, _1, acc_value_cnn = sess.run([loss_cnn, grad_update, sv.global_step,
                                                                     metrics_op_cnn, accuracy_streaming_cnn],
                                                                    feed_dict={images: batch_x, labels: batch_y})

                logging.info("At step %d/%d, loss= %.4f, accuracy=%.2f;",
                             step, int(num_steps_per_epoch * CNN2TrainConfig.num_epochs),
                             loss_cnn_value, 100*acc_value_cnn)
                if step % 100==0:
                    logging.info("Saving model as at step%500")
                    sv.saver.save(sess, sv.save_path, global_step=step)

            # Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step=step)
            self.finished.emit()

    @pyqtSlot()
    def stop_call(self):
        """Stop training process and exit"""
        print("Stopping Training..")
        self.epoch.emit(0)
        self.finished.emit()