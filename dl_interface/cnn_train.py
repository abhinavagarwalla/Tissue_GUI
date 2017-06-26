import shutil
from time import time
import os

import tensorflow as tf
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import random
import glob

from nets import nets_factory
from preprocessing import preprocessing_factory
from interface.image_slide import ImageClass

from dl_interface.model_config import CNN2TrainConfig
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import pickle

import numpy as np


slim = tf.contrib.slim

class DataIter():
    def __init__(self):
        self.wsi_list = glob.glob(CNN2TrainConfig.DATA_IMAGES_PATH + os.sep + '*')
        self.images_list = []
        for i in self.wsi_list:
            self.images_list.extend(glob.glob(i + os.sep + '*'))
        random.shuffle(self.images_list)
        self.num_samples = len(self.images_list)
        self.iter = 0
        self.preprocessor = preprocessing_factory.get_preprocessing_fn(name='camelyon')

    def next_batch(self):
        x = []
        y = []
        prev_wsi_id = 0
        while len(x) != CNN2TrainConfig.batch_size:
            patch_id = self.images_list[self.iter]
            wsi_id = patch_id.split(os.sep)[-2]
            if prev_wsi_id!=wsi_id:
                self.wsi_obj = ImageClass(CNN2TrainConfig.WSI_BASE_PATH + os.sep + wsi_id + '.tif')
            coors = patch_id.split('(')[1].split(')')[0]
            w, h = list(map(int, coors.split(',')))
            im = np.array(
                self.wsi_obj.read_region((w, h), 0, (CNN2TrainConfig.PATCH_SIZE * CNN2TrainConfig.IMAGE_SIZE,
                                                      CNN2TrainConfig.PATCH_SIZE * CNN2TrainConfig.IMAGE_SIZE)).convert('RGB'))

            for i in range(0, CNN2TrainConfig.IMAGE_SIZE * CNN2TrainConfig.PATCH_SIZE, CNN2TrainConfig.IMAGE_SIZE):
                for j in range(0, CNN2TrainConfig.IMAGE_SIZE * CNN2TrainConfig.PATCH_SIZE, CNN2TrainConfig.IMAGE_SIZE):
                    img = im[i:i + CNN2TrainConfig.IMAGE_SIZE, j:j + CNN2TrainConfig.IMAGE_SIZE]
                    x.append(img)
                    # x.append(self.preprocessor.preprocess_image(tf.convert_to_tensor(img),
                    #                                             CNN2TrainConfig.IMAGE_SIZE, CNN2TrainConfig.IMAGE_SIZE,
                    #                                             CNN2TrainConfig.IMAGE_SIZE, CNN2TrainConfig.IMAGE_SIZE,
                    #                                             is_training=True))

            # im = np.array(im).reshape(CNN2TrainConfig.IMAGE_SIZE, CNN2TrainConfig.IMAGE_SIZE, 3)
            # feat = pickle.load(open(self.images_list[self.iter], 'rb'))
            # wsi_name = self.images_list[self.iter].split(os.sep)[-1]
            label = np.load(CNN2TrainConfig.DATA_LABELS_PATH + os.sep + patch_id.split(os.sep)[-1].split('_(')[0]
                            + os.sep + patch_id.split(os.sep)[-1].replace('features.pkl', 'label.npy')).reshape(-1)
            label_inv = 1-label
            label = np.vstack((label, label_inv)).T
            y.extend(label)
            self.iter += 1
            if self.iter == self.num_samples:
                self.iter = 0
        return x, y

class CNN2Train(QObject):
    finished = pyqtSignal()
    epoch = pyqtSignal(int)

    def initialize(self):
        self.t0 = time()
        self.dataloader = DataIter()

    @pyqtSlot()
    def train(self):
        # Saver and initialisation
        print("starting training")
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
        accuracy_batch_cnn, accuracy_batch_cnn_update = tf.metrics.accuracy(labels_class, cnn_preds_class)
        precision_batch_cnn, precision_batch_cnn_update = tf.metrics.precision(labels_class, cnn_preds_class)
        recall_batch_cnn, recall_batch_cnn_update = tf.metrics.recall(labels_class, cnn_preds_class)
        metrics_op_cnn = tf.group(recall_streaming_cnn_update, precision_streaming_cnn_update, accuracy_streaming_cnn_update,
                              recall_batch_cnn_update, precision_batch_cnn_update, accuracy_batch_cnn_update)

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

        # predictions = tf.reshape(model_out, [-1])
        # labels_flat = tf.reshape(labels, [-1])

        # Now finally create all the summaries you need to monitor and group them into one summary op.
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
            return saver_all.restore(sess, CNN2TrainConfig.checkpoint_file)

        # Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=CNN2TrainConfig.log_dir, summary_op=None, init_fn=restore_fn, saver=saver_all)  # restore_fn)

        logging.info("now starting session")
        # Run the managed session
        with sv.managed_session() as sess:
            logging.info("initialiser run")
            for step in range(int(num_steps_per_epoch * CNN2TrainConfig.num_epochs)):
                batch_x, batch_y = self.dataloader.next_batch()

                # Log the summaries every 10 step.
                if step % 100 == 0:
                    loss_cnn_value, _, summaries,\
                    global_step_count, _1, acc_value_cnn = sess.run([loss_cnn, grad_update, my_summary_op,
                                sv.global_step,  metrics_op_cnn, accuracy_batch_cnn],
                                feed_dict={images: batch_x, labels: batch_y})
                    sv.summary_computed(sess, summaries, global_step=step)
                else:
                    loss_cnn_value, _, \
                    global_step_count, _1, acc_value_cnn = sess.run([loss_cnn, grad_update, sv.global_step,
                                                                     metrics_op_cnn, accuracy_batch_cnn],
                                                                    feed_dict={images: batch_x, labels: batch_y})

                logging.info("At step %d/%d, loss= %.4f, accuracy=%.2f;",
                             step, int(num_steps_per_epoch * CNN2TrainConfig.num_epochs),
                             loss_cnn_value, 100*acc_value_cnn)
                if step % 500==0:
                    logging.info("Saving model as at step%500")
                    sv.saver.save(sess, sv.save_path, global_step=step)

            # Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            sv.saver.save(sess, sv.save_path, global_step=step)
            self.finished.emit()

    @pyqtSlot()
    def stop_call(self):
        print("Stopping Training..")
        self.epoch.emit(0)
        self.finished.emit()