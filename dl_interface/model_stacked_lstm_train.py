# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation for training 2D-LSTM"""
from time import time
import os
import tensorflow as tf
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from dl_interface.model_config import LSTMTrainConfig
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from dataio.lstm_batch_iter import LSTMTrainPNDataIter, LSTMTrainMatDataIter
from nets import nets_factory

slim = tf.contrib.slim

class StackedLSTMTrain(QObject):
    finished = pyqtSignal()
    epoch = pyqtSignal(int)

    def initialize(self):
        self.t0 = time()
        # self.dataloader = LSTMTrainPNDataIter()
        self.dataloader = LSTMTrainMatDataIter()

    @pyqtSlot()
    def train(self):
        """Starts training process"""
        # Saver and initialisation
        print("starting stacked training")
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

        # Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = self.dataloader.num_samples / LSTMTrainConfig.batch_size
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        decay_steps = int(LSTMTrainConfig.num_epochs_before_decay * num_steps_per_epoch)


        images = tf.placeholder(tf.float32, [LSTMTrainConfig.batch_size, LSTMTrainConfig.PATCH_SIZE,
                                             LSTMTrainConfig.PATCH_SIZE, LSTMTrainConfig.CHANNELS])
        labels = tf.placeholder(tf.float32, [LSTMTrainConfig.batch_size, LSTMTrainConfig.PATCH_SIZE,
                                             LSTMTrainConfig.PATCH_SIZE, 1])#
        cnn_preds = tf.placeholder(tf.float32, [LSTMTrainConfig.batch_size, LSTMTrainConfig.PATCH_SIZE,
                                             LSTMTrainConfig.PATCH_SIZE, LSTMTrainConfig.NUM_CLASSES])
        model_out = nets_factory.get_network_fn('Stacked-2D-LSTM-8c', images, num_classes=LSTMTrainConfig.NUM_CLASSES,
                                                is_training=True)

        logging.info("Stacking out input")
        model_out_flat = tf.reshape(model_out, shape=(-1, LSTMTrainConfig.NUM_CLASSES))

        non_tumor_label = tf.subtract(tf.ones((LSTMTrainConfig.PATCH_SIZE, LSTMTrainConfig.PATCH_SIZE, 1)), labels)
        combined_label = tf.concat([non_tumor_label, labels], axis=3)
        labels_flat = tf.reshape(combined_label, shape=(-1,LSTMTrainConfig.NUM_CLASSES))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model_out_flat,
                                                                  labels=labels_flat))

        cnn_preds_flat = tf.reshape(cnn_preds, shape=(-1, LSTMTrainConfig.NUM_CLASSES))
        loss_cnn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cnn_preds_flat,
                                                                  labels=labels_flat))

        model_out_class = tf.argmax(model_out_flat, axis=1)
        cnn_preds_class = tf.argmax(cnn_preds_flat, axis=1)
        labels_class = tf.argmax(labels_flat, axis=1)

        accuracy_streaming, accuracy_streaming_update = tf.contrib.metrics.streaming_accuracy(model_out_class, labels_class)
        precision_streaming, precision_streaming_update = tf.contrib.metrics.streaming_precision(model_out_class, labels_class)
        recall_streaming, recall_streaming_update = tf.contrib.metrics.streaming_recall(model_out_class, labels_class)
        accuracy_batch, accuracy_batch_update = tf.metrics.accuracy(labels_class, model_out_class)
        precision_batch, precision_batch_update = tf.metrics.precision(labels_class, model_out_class)
        recall_batch, recall_batch_update = tf.metrics.recall(labels_class, model_out_class)
        # metrics_op = tf.group(recall_streaming_update, precision_streaming_update, accuracy_streaming_update,
        #                       recall_batch_update, precision_batch_update, accuracy_batch_update)

        accuracy_streaming_cnn, accuracy_streaming_cnn_update = tf.contrib.metrics.streaming_accuracy(cnn_preds_class,
                                                                                              labels_class)
        precision_streaming_cnn, precision_streaming_cnn_update = tf.contrib.metrics.streaming_precision(cnn_preds_class,
                                                                                                 labels_class)
        recall_streaming_cnn, recall_streaming_cnn_update = tf.contrib.metrics.streaming_recall(cnn_preds_class, labels_class)
        accuracy_batch_cnn, accuracy_batch_cnn_update = tf.metrics.accuracy(labels_class, cnn_preds_class)
        precision_batch_cnn, precision_batch_cnn_update = tf.metrics.precision(labels_class, cnn_preds_class)
        recall_batch_cnn, recall_batch_cnn_update = tf.metrics.recall(labels_class, cnn_preds_class)
        metrics_op = tf.group(recall_streaming_update, precision_streaming_update, accuracy_streaming_update,
                              recall_batch_update, precision_batch_update, accuracy_batch_update,
                                  recall_streaming_cnn_update, precision_streaming_cnn_update, accuracy_streaming_cnn_update,
                              recall_batch_cnn_update, precision_batch_cnn_update, accuracy_batch_cnn_update)

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        # Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate=LSTMTrainConfig.initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=LSTMTrainConfig.learning_rate_decay_factor,
            staircase=True)

        # Now we can define the optimizer that takes on the learning rate
        grad_update = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        # predictions = tf.reshape(model_out, [-1])
        # labels_flat = tf.reshape(labels, [-1])

        # Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Absolute_L1_Loss', loss)
        tf.summary.scalar('accuracy_streaming', accuracy_streaming)
        tf.summary.scalar('precision_streaming', precision_streaming)
        tf.summary.scalar('recall_streaming', recall_streaming)
        tf.summary.scalar('accuracy_batch', accuracy_batch)
        tf.summary.scalar('precision_batch', precision_batch)
        tf.summary.scalar('recall_batch', recall_batch)

        tf.summary.scalar('losses/Absolute_L1_Loss_CNN', loss_cnn)
        tf.summary.scalar('accuracy_streaming_cnn', accuracy_streaming_cnn)
        tf.summary.scalar('precision_streaming_cnn', precision_streaming_cnn)
        tf.summary.scalar('recall_streaming_cnn', recall_streaming_cnn)
        tf.summary.scalar('accuracy_batch_cnn', accuracy_batch_cnn)
        tf.summary.scalar('precision_batch_cnn', precision_batch_cnn)
        tf.summary.scalar('recall_batch_cnn', recall_batch_cnn)

        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()

        # Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        saver_all = tf.train.Saver(max_to_keep=None)

        def restore_fn(sess):
            return saver_all.restore(sess, LSTMTrainConfig.checkpoint_file)

        # Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=LSTMTrainConfig.log_dir, summary_op=None, init_fn=None, saver=saver_all)  # restore_fn)

        logging.info("now starting session")
        # Run the managed session
        # with tf.Session() as sess:
        with sv.managed_session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.60))) as sess:
            # writer = tf.summary.FileWriter(LSTMTrainConfig.log_dir, sess.graph)
            # sess.run(tf.global_variables_initializer())
            logging.info("initialiser run")
            for step in range(int(num_steps_per_epoch * LSTMTrainConfig.num_epochs)):
                batch_x, batch_y, cnn_y, cnn_logits = self.dataloader.next_batch()

                # Log the summaries every 10 step.
                if step % 100 == 0:
                    loss_value, loss_cnn_value, _, summaries,\
                    global_step_count, _1, acc_value, acc_value_cnn = sess.run([loss, loss_cnn, grad_update, my_summary_op,
                                sv.global_step, metrics_op, accuracy_batch, accuracy_batch_cnn],
                                feed_dict={images: batch_x, labels: batch_y, cnn_preds: cnn_logits})
                    # summaries = sess.run(my_summary_op)
                    sv.summary_computed(sess, summaries, global_step=step)
                    # writer.add_summary(summaries, step)
                else:
                    loss_value, loss_cnn_value, _, \
                    global_step_count, _1, acc_value, acc_value_cnn = sess.run([loss, loss_cnn, grad_update,
                                                                     sv.global_step, metrics_op,
                                                                     accuracy_batch, accuracy_batch_cnn],
                                                                    feed_dict={images: batch_x, labels: batch_y,
                                                                               cnn_preds: cnn_logits})

                logging.info("At step %d/%d, loss= %.4f, accuracy=%.2f; cnn_only_loss= %.4f, cnn_only_accuracy=%.2f",
                             step, int(num_steps_per_epoch * LSTMTrainConfig.num_epochs),
                             loss_value, 100*acc_value, loss_cnn_value, 100*acc_value_cnn)
                if step % 200==0:
                    logging.info("Saving model as at step%500")
                    # saver.save(sess, LSTMTrainConfig.log_dir + os.sep + "lstm_model", global_step=step)
                    sv.saver.save(sess, sv.save_path, global_step=step)
            # writer.close()
            # Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            # saver.save(sess, LSTMTrainConfig.log_dir + os.sep + "lstm_model",global_step=step)
            sv.saver.save(sess, sv.save_path, global_step=step)
            self.finished.emit()

    @pyqtSlot()
    def stop_call(self):
        """Stop training process and exit thread"""
        print("Stopping Training..")
        self.epoch.emit(0)
        self.finished.emit()