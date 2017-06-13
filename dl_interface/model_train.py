import shutil
from time import time
import os

import tensorflow as tf
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from dl_interface.model_config import TrainConfig
from nets import nets_factory
from preprocessing import preprocessing_factory

from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging


slim = tf.contrib.slim


class Train(QObject):
    finished = pyqtSignal()
    epoch = pyqtSignal(int)

    def initialize(self):
        self.t0 = time()
        # self.images_test = tf.placeholder(tf.float32, shape=(None, Config.PATCH_SIZE, Config.PATCH_SIZE, 3))
        # self.output = nets_factory.get_network_fn(name='unet', images=self.images_test, is_training=False)
        # self.dataloader = Data(preprocessor='stain_norm', outshape=self.output.shape)
        self.preprocessor = preprocessing_factory.get_preprocessing_fn(name='camelyon')

        labels = open(TrainConfig.labels_file, 'r')

        # Create a dictionary to refer each label to their string name
        self.labels_to_name = {}
        for line in labels:
            label, string_name = line.split(':')
            string_name = string_name[:-1]  # Remove newline
            self.labels_to_name[int(label)] = string_name

        # Create the file pattern of your TFRecord files so that it could be recognized later on
        # self.file_pattern = 'flowers_%s_*.tfrecord'

        # Create a dictionary that will help people understand your dataset better. This is required by the Dataset class later.
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
        file_pattern_path = os.path.join(TrainConfig.dataset_dir, TrainConfig.file_pattern % (split_name))

        # Count the total number of examples in all of these shard
        num_samples = 188206 #0
        # file_pattern_for_counting = 'Camelyon_tfr_' + split_name
        # tfrecords_to_count = [os.path.join(TrainConfig.dataset_dir, file) for file in os.listdir(TrainConfig.dataset_dir) if
        #                       file.startswith(file_pattern_for_counting)]
        # for tfrecord_file in tfrecords_to_count:
        #     for record in tf.python_io.tf_record_iterator(tfrecord_file):
        #         num_samples += 1

        # Create a reader, which must be a TFRecord reader in this case
        reader = tf.TFRecordReader

        # Create the keys_to_features dictionary for the decoder
        keys_to_features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='PNG'),
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
            num_classes=TrainConfig.num_classes,
            labels_to_name=labels_to_name_dict,
            items_to_descriptions=self.items_to_descriptions)

        return dataset

    def load_batch(self, dataset, batch_size, height=TrainConfig.image_size, width=TrainConfig.image_size, is_training=True):
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
        image = self.preprocessor.preprocess_image(raw_image, height, width, TrainConfig.source_size, TrainConfig.source_size,is_training)

        # As for the raw images, we just do a simple reshape to batch it up
        raw_image = tf.expand_dims(raw_image, 0)
        raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
        raw_image = tf.squeeze(raw_image)

        # Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
        images, raw_images, labels = tf.train.shuffle_batch(
            [image, raw_image, label],
            batch_size=batch_size,
            num_threads=4,
            capacity=5 * batch_size,
            allow_smaller_final_batch=True,
            enqueue_many=True,
            min_after_dequeue=2 * batch_size)

        return images, raw_images, labels

    @pyqtSlot()
    def train(self):
        # Saver and initialisation
        print("starting training")
        self.initialize()
        # saver = tf.train.Saver()
        self.epoch.emit(0)
        if not os.path.exists(TrainConfig.log_dir):
            os.mkdir(TrainConfig.log_dir)

        # ======================= TRAINING PROCESS =========================
        # Now we start to construct the graph and build our model
        with tf.Graph().as_default() as graph:
            tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

            # First create the dataset and load one batch
            dataset = self.get_split('train')
            images, _, labels = self.load_batch(dataset, batch_size=TrainConfig.batch_size)

            # Know the number steps to take before decaying the learning rate and batches per epoch
            num_batches_per_epoch = dataset.num_samples / TrainConfig.batch_size
            num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
            decay_steps = int(TrainConfig.num_epochs_before_decay * num_steps_per_epoch)

            # Create the model inference
            logits, end_points = nets_factory.get_network_fn(name='inception_resnet_v2', images=images,
                                                             num_classes=dataset.num_classes, is_training=True)

            # with slim.arg_scope(inception_resnet_v2_arg_scope()):
            #     logits, end_points = inception_resnet_v2(images, num_classes=dataset.num_classes, is_training=True)

            # Define the scopes that you want to exclude for restoration
            exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
            variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

            # Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
            one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

            # Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
            loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
            total_loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

            # Create the global step for monitoring the learning_rate and training.
            global_step = get_or_create_global_step()

            # Define your exponentially decaying learning rate
            lr = tf.train.exponential_decay(
                learning_rate=TrainConfig.initial_learning_rate,
                global_step=global_step,
                decay_steps=decay_steps,
                decay_rate=TrainConfig.learning_rate_decay_factor,
                staircase=True)

            # Now we can define the optimizer that takes on the learning rate
            optimizer = tf.train.AdamOptimizer(learning_rate=lr)

            # Create the train_op.
            train_op = slim.learning.create_train_op(total_loss, optimizer)

            # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
            predictions = tf.argmax(end_points['Predictions'], 1)
            probabilities = end_points['Predictions']
            accuracy_streaming, accuracy_streaming_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
            precision_streaming, precision_streaming_update = tf.contrib.metrics.streaming_precision(predictions, labels)
            recall_streaming, recall_streaming_update = tf.contrib.metrics.streaming_recall(predictions, labels)
            accuracy_batch, accuracy_batch_update = tf.metrics.accuracy(labels, predictions)
            precision_batch, precision_batch_update = tf.metrics.precision(labels, predictions)
            recall_batch, recall_batch_update = tf.metrics.recall(labels, predictions)
            metrics_op = tf.group(recall_streaming_update, precision_streaming_update, accuracy_streaming_update,
                                  recall_batch_update, precision_batch_update, accuracy_batch_update, probabilities)

            # Now finally create all the summaries you need to monitor and group them into one summary op.
            tf.summary.scalar('losses/Total_Loss', total_loss)
            tf.summary.scalar('accuracy_streaming', accuracy_streaming)
            tf.summary.scalar('precision_streaming', precision_streaming)
            tf.summary.scalar('recall_streaming', recall_streaming)
            tf.summary.scalar('accuracy_batch', accuracy_batch)
            tf.summary.scalar('precision_batch', precision_batch)
            tf.summary.scalar('recall_batch', recall_batch)
            tf.summary.scalar('learning_rate', lr)
            my_summary_op = tf.summary.merge_all()

            # Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
            def train_step(sess, train_op, global_step):
                '''
                Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
                '''
                # Check the time for each sess run
                start_time = time()
                total_loss, global_step_count, _, acc_str, pre_str, rec_str, acc_bat, pre_bat, rec_bat = sess.run([train_op, global_step, metrics_op,
                                                                            accuracy_streaming, precision_streaming, recall_streaming,
                                                                            accuracy_batch, precision_batch, recall_batch])
                time_elapsed = time() - start_time

                # Run the logging to print some results
                logging.info('global step %s: loss: %.4f (%.2f sec/step) accuracy_streaming=%.4f,'
                             ' precision_streaming=%.4f, recall_streaming=%.4f ;;'
                             'accuracy_batch=%.4f, precision_batch=%.4f, recall_batch=%.4f', global_step_count,
                             total_loss, time_elapsed, acc_str, pre_str, rec_str, acc_bat, pre_bat, rec_bat)

                return total_loss, global_step_count

            # Now we create a saver function that actually restores the variables from a checkpoint file in a sess
            saver = tf.train.Saver(variables_to_restore)

            def restore_fn(sess):
                return saver.restore(sess, TrainConfig.checkpoint_file)

            # Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
            sv = tf.train.Supervisor(logdir=TrainConfig.log_dir, summary_op=None, init_fn=restore_fn)

            # Run the managed session
            with sv.managed_session() as sess:
                for step in range(int(num_steps_per_epoch * TrainConfig.num_epochs)):
                    logging.info("Another step")
                    # for step in xrange(1):
                    # At the start of every epoch, show the vital information:
                    if step % num_batches_per_epoch == 0:
                        logging.info('Epoch %s/%s', step / num_batches_per_epoch + 1, TrainConfig.num_epochs)
                        learning_rate_value, accuracy_value = sess.run([lr, accuracy_streaming])
                        logging.info('Current Learning Rate: %s', learning_rate_value)
                        logging.info('Current Streaming Accuracy: %s', accuracy_value)

                        # optionally, print your logits and predictions for a sanity check that things are going fine.
                        # logits_value, probabilities_value, predictions_value, labels_value = sess.run(
                        #     [logits, probabilities, predictions, labels])
                        # print
                        # 'logits: \n', logits_value
                        # print
                        # 'Probabilities: \n', probabilities_value
                        # print
                        # 'predictions: \n', predictions_value
                        # print
                        # 'Labels:\n:', labels_value

                    # Log the summaries every 10 step.
                    if step % 100 == 0:
                        loss, _ = train_step(sess, train_op, sv.global_step)
                        summaries = sess.run(my_summary_op)
                        sv.summary_computed(sess, summaries)

                    # If not, simply run the training step
                    else:
                        loss, _ = train_step(sess, train_op, sv.global_step)

                    if step % 500==0:
                        sv.saver.save(sess, sv.save_path, global_step=sv.global_step)
                # We log the final training loss and accuracy
                logging.info('Final Loss: %s', loss)
                logging.info('Final Accuracy: %s', sess.run(accuracy_streaming))

                # Once all the training has been done, save the log files and checkpoint model
                logging.info('Finished training! Saving model to disk now.')
                # saver.save(sess, "./flowers_model.ckpt")
                sv.saver.save(sess, sv.save_path, global_step=sv.global_step)
            self.finished.emit()

    @pyqtSlot()
    def stop_call(self):
        print("Stopping Training..")
        # if os.path.exists(TrainConfig.RESULT_PATH):
        #     try:
        #         shutil.rmtree(TrainConfig.RESULT_PATH)
        #         print("Result tree removed")
        #     except:
        #         pass
        self.epoch.emit(0)
        self.finished.emit()