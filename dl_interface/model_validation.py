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
import glob

import tensorflow as tf
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from dl_interface.model_config import ValidConfig
from nets import nets_factory
from preprocessing import preprocessing_factory

from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging


slim = tf.contrib.slim


class Validate(QObject):
    finished = pyqtSignal()
    epoch = pyqtSignal(int)

    def initialize(self):
        self.t0 = time()
        # self.images_test = tf.placeholder(tf.float32, shape=(None, Config.PATCH_SIZE, Config.PATCH_SIZE, 3))
        # self.output = nets_factory.get_network_fn(name='unet', images=self.images_test, is_training=False)
        # self.dataloader = Data(preprocessor='stain_norm', outshape=self.output.shape)
        self.preprocessor = preprocessing_factory.get_preprocessing_fn(name='camelyon')

        labels = open(ValidConfig.labels_file, 'r')
        #
        # # Create a dictionary to refer each label to their string name
        self.labels_to_name = {}
        for line in labels:
            label, string_name = line.split(':')
            string_name = string_name[:-1]  # Remove newline
            self.labels_to_name[int(label)] = string_name
        #
        #
        # # Create a dictionary that will help people understand your dataset better. This is required by the Dataset class later.
        self.items_to_descriptions = {
            'image': 'A 3-channel RGB coloured flower image that is either tumor or normal.',
            'label': 'A label that is as such -- 0:normal, 1:tumor'
        }

    def get_split(self, split_name):
        '''
        Obtains the split - training or validation - to create a Dataset class for feeding the examples into a queue later on. This function will
        set up the decoder and dataset information all into one Dataset class so that you can avoid the brute work later on.
        Your file_pattern is very important in locating the files later. 

        INPUTS:
        - split_name(str): 'train' or 'validation'. Used to get the correct data split of tfrecord files
        - dataset_dir(str): the dataset directory where the tfrecord files are located
        - file_pattern(str): the file name structure of the tfrecord files in order to get the correct data

        OUTPUTS:
        - dataset (Dataset): A Dataset class object where we can read its various components for easier batch creation later.
        '''

        # First check whether the split_name is train or validation
        if split_name not in ['train', 'validation']:
            raise ValueError(
                'The split_name %s is not recognized. Please input either train or validation as the split_name' % (
                    split_name))

        # Create the full path for a general file_pattern to locate the tfrecord_files
        file_pattern_path = os.path.join(ValidConfig.dataset_dir, ValidConfig.file_pattern % (split_name))

        # Count the total number of examples in all of these shard
        num_samples = 79078 #0
        # file_pattern_for_counting = 'Camelyon_tfr_' + split_name
        # tfrecords_to_count = [os.path.join(ValidConfig.dataset_dir, file) for file in os.listdir(ValidConfig.dataset_dir) if
        #                       file.startswith(file_pattern_for_counting)]
        # for tfrecord_file in tfrecords_to_count:
        #     for record in tf.python_io.tf_record_iterator(tfrecord_file):
        #         num_samples += 1
        # print(num_samples)

        # Create a reader, which must be a TFRecord reader in this case
        reader = tf.TFRecordReader

        # Create the keys_to_features dictionary for the decoder
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
            'image/class/label': tf.FixedLenFeature(
                [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        }

        # Create the items_to_handlers dictionary for the decoder.
        items_to_handlers = {
            'image': slim.tfexample_decoder.Image(),
            'label': slim.tfexample_decoder.Tensor('image/class/label'),
        }

        # Start to create the decoder
        decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

        # Create the labels_to_name file
        labels_to_name_dict = self.labels_to_name

        # Actually create the dataset
        dataset = slim.dataset.Dataset(
            data_sources=file_pattern_path,
            decoder=decoder,
            reader=reader,
            num_readers=4,
            num_samples=num_samples,
            num_classes=ValidConfig.num_classes,
            labels_to_name=labels_to_name_dict,
            items_to_descriptions=self.items_to_descriptions)

        return dataset

    def load_batch(self, dataset, batch_size, height=ValidConfig.image_size, width=ValidConfig.image_size, is_training=True):
        '''
        Loads a batch for training.

        INPUTS:
        - dataset(Dataset): a Dataset class object that is created from the get_split function
        - batch_size(int): determines how big of a batch to train
        - height(int): the height of the image to resize to during preprocessing
        - width(int): the width of the image to resize to during preprocessing
        - is_training(bool): to determine whether to perform a training or evaluation preprocessing

        OUTPUTS:
        - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
        - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).

        '''
        # First create the data_provider object
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            common_queue_capacity=24 + 3 * batch_size,
            common_queue_min=24)

        # Obtain the raw image using the get method
        raw_image, label = data_provider.get(['image', 'label'])

        # Perform the correct preprocessing for this image depending if it is training or evaluating
        # image = inception_preprocessing.preprocess_image(raw_image, height, width, is_training)
        image = self.preprocessor.preprocess_image(raw_image, height, width, ValidConfig.source_size, ValidConfig.source_size,is_training)

        # As for the raw images, we just do a simple reshape to batch it up
        raw_image = tf.expand_dims(raw_image, 0)
        raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
        raw_image = tf.squeeze(raw_image)

        # Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
        images, raw_images, labels = tf.train.shuffle_batch(
            [image, raw_image, label],
            batch_size = batch_size,
            num_threads = 4,
            capacity = 5 * batch_size,
            allow_smaller_final_batch = True,
            min_after_dequeue = 2*batch_size)

        return images, raw_images, labels

    @pyqtSlot()
    def run(self):
        mlist = glob.glob(ValidConfig.log_dir + os.sep + "model.ckpt-*.meta")
        mlist = [i[:-5] for i in mlist]
        for i in range(len(mlist)):
            print(mlist[i])
            ValidConfig.checkpoint_file = mlist[i]
            self.run_once()

    @pyqtSlot()
    def run_once(self):
        # Saver and initialisation
        print("started validation")
        self.initialize()
        # saver = tf.train.Saver()
        self.epoch.emit(0)

        # Create log_dir for evaluation information
        if not os.path.exists(ValidConfig.log_eval):
            os.mkdir(ValidConfig.log_eval)

        # Just construct the graph from scratch again
        with tf.Graph().as_default() as graph:
            tf.logging.set_verbosity(tf.logging.INFO)
            # Get the dataset first and load one batch of validation images and labels tensors.
            # Set is_training as False so as to use the evaluation preprocessing
            dataset = self.get_split('validation')
            images, raw_images, labels = self.load_batch(dataset, batch_size=ValidConfig.batch_size, is_training=False)

            # Create some information about the training steps
            num_batches_per_epoch = dataset.num_samples / ValidConfig.batch_size
            num_steps_per_epoch = num_batches_per_epoch

            # Create the model inference
            # logits, end_points = nets_factory.get_network_fn(name='inception_resnet_v2', images=images,
            #                                                  num_classes=dataset.num_classes, is_training=False)

            logits, end_points = nets_factory.get_network_fn(name='alexnet', images=images,
                                                             num_classes=dataset.num_classes, is_training=True)

            # #get all the variables to restore from the checkpoint file and create the saver function to restore
            variables_to_restore = slim.get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)

            def restore_fn(sess):
                return saver.restore(sess, ValidConfig.checkpoint_file)

            # Just define the metrics to track without the loss or whatsoever
            # predictions = tf.argmax(end_points['Predictions'], 1)
            predictions = tf.argmax(end_points['alexnet_v2/fc8'], 1)
            accuracy_streaming, accuracy_streaming_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
            precision_streaming, precision_streaming_update = tf.contrib.metrics.streaming_precision(predictions,
                                                                                                     labels)
            recall_streaming, recall_streaming_update = tf.contrib.metrics.streaming_recall(predictions, labels)
            accuracy_batch, accuracy_batch_update = tf.metrics.accuracy(labels, predictions)
            precision_batch, precision_batch_update = tf.metrics.precision(labels, predictions)
            recall_batch, recall_batch_update = tf.metrics.recall(labels, predictions)
            metrics_op = tf.group(recall_streaming_update, precision_streaming_update, accuracy_streaming_update,
                                  recall_batch_update, precision_batch_update, accuracy_batch_update)

            # Create the global step and an increment op for monitoring
            global_step = get_or_create_global_step()
            global_step_op = tf.assign(global_step,
                                       global_step + 1)  # no apply_gradient method so manually increasing the global_step

            # Create a evaluation step function
            def eval_step(sess, metrics_op, global_step):
                '''
                Simply takes in a session, runs the metrics op and some logging information.
                '''
                start_time = time()
                _, global_step_count, acc_str, pre_str, rec_str, acc_bat, pre_bat, rec_bat = sess.run([metrics_op,
                                                            global_step_op, accuracy_streaming, precision_streaming,
                                                            recall_streaming, accuracy_batch, precision_batch, recall_batch])
                time_elapsed = time() - start_time

                # Log some information
                logging.info('Global Step %s: Streaming Accuracy: %.4f, Precision: %.4f, Recall: %.4f (%.2f sec/step)'
                             'Batch: Accuracy: %.4f, Precision: %.4f, Recall: %.4f',
                             global_step_count, acc_str, pre_str, rec_str, time_elapsed, acc_bat, pre_bat, rec_bat)

                return acc_str

            # Define some scalar quantities to monitor
            tf.summary.scalar('accuracy_streaming', accuracy_streaming)
            tf.summary.scalar('precision_streaming', precision_streaming)
            tf.summary.scalar('recall_streaming', recall_streaming)
            tf.summary.scalar('accuracy_batch', accuracy_batch)
            tf.summary.scalar('precision_batch', precision_batch)
            tf.summary.scalar('recall_batch', recall_batch)
            my_summary_op = tf.summary.merge_all()

            # Get your supervisor
            sv = tf.train.Supervisor(logdir=ValidConfig.log_eval, summary_op=None, saver=None, init_fn=restore_fn)

            # Now we are ready to run in one session
            with sv.managed_session() as sess:
                for step in range(int(num_steps_per_epoch * ValidConfig.num_epochs)):
                    sess.run(sv.global_step)
                    # print vital information every start of the epoch as always
                    if step % num_batches_per_epoch == 0:
                        logging.info('Epoch: %s/%s', step / num_batches_per_epoch + 1, ValidConfig.num_epochs)
                        logging.info('Current Streaming Accuracy: %.4f', sess.run(accuracy_streaming))

                    # Compute summaries every 10 steps and continue evaluating
                    if step % 10 == 0:
                        eval_step(sess, metrics_op=metrics_op, global_step=sv.global_step)
                        summaries = sess.run(my_summary_op)
                        sv.summary_computed(sess, summaries)

                        # optionally, print your logits and predictions for a sanity check that things are going fine.
                        logits_value, predictions_value, labels_value = sess.run(
                            [logits, predictions, labels])
                        # print('logits: \n', logits_value)
                        print('predictions: \n', predictions_value)
                        print('Labels:\n:', labels_value)
                    # Otherwise just run as per normal
                    else:
                        eval_step(sess, metrics_op=metrics_op, global_step=sv.global_step)

                # At the end of all the evaluation, show the final accuracy
                logging.info('Final Streaming Accuracy: %.4f', sess.run(accuracy_streaming))
                logging.info(
                    'Model evaluation has completed for one checkpoint')
            self.finished.emit()

    @pyqtSlot()
    def stop_call(self):
        print("Stopping Validation..")
        self.epoch.emit(0)
        self.finished.emit()