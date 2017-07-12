# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generates and saves intermediate layer features of a CNN"""
from time import time
import os

import tensorflow as tf
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import numpy as np
import glob

from dl_interface.model_config import LSTMDataConfig
from nets import nets_factory
from interface.image_slide import ImageClass
from dataio.lstm_batch_iter import LSTMDataMod

class TestLSTMSave(QObject):
    """Saves intermediate layer features for CNN in a directorys"""
    finished = pyqtSignal()
    epoch = pyqtSignal(int)

    def initialize(self):
        self.t0 = time()
        self.images_test = tf.placeholder(tf.float32, shape=(None, LSTMDataConfig.PATCH_SIZE, LSTMDataConfig.PATCH_SIZE, 3))
        self.logits, self.end_points = nets_factory.get_network_fn(name='alexnet', images=self.images_test,
                                                         num_classes=LSTMDataConfig.NUM_CLASSES, is_training=False)

    def init_data_loader(self):
        """Initialise dataloader from dataio module"""
        self.dataloader = LSTMDataMod(preprocessor='camelyon')

        # if os.path.exists(LSTMDataConfig.RESULT_PATH):
        #     shutil.rmtree(Config.RESULT_PATH)
        # os.mkdir(Config.RESULT_PATH)
        # os.mkdir(Config.RESULT_PATH + os.sep + "predictions_png")

    @pyqtSlot()
    def test(self):
        """Start generating features for a list of WSIs"""
        self.wsi_iter = None
        self.classes = os.listdir(LSTMDataConfig.WSI_FOLDER_PATH)
        # self.wsi_list = dict((i, os.listdir(PatchConfig.WSI_FOLDER_PATH+os.sep+i)) for i in self.classes)
        self.wsi_list = dict((i, glob.glob(LSTMDataConfig.WSI_FOLDER_PATH + os.sep + i + os.sep + "*.tif")) for i in self.classes)
        self.classes_dict = dict((i, self.classes[i]) for i in range(len(self.classes)))
        self.initialize()
        # mlist = [i[:-5] for i in mlist]
        for i in range(len(self.wsi_list["Tumor"])):
            print(self.wsi_list["Tumor"][i])
            self.wsi_iter = i
            LSTMDataConfig.WSI_PATH = self.wsi_list["Tumor"][self.wsi_iter]
            LSTMDataConfig.MASK_PATH = LSTMDataConfig.WSI_FOLDER_PATH + os.sep + "Tumor" + os.sep + "Mask_Tissue" + os.sep +\
                             self.wsi_list["Tumor"][self.wsi_iter].split(os.sep)[-1]
            self.test_once()
        print("Finished for all assigned WSIs")

    @pyqtSlot()
    def test_once(self):
        """Generates and saves features for a particular WSI"""
        # Saver and initialisation
        self.init_data_loader()
        saver = tf.train.Saver()
        self.epoch.emit(0)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.20)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            saver.restore(sess, LSTMDataConfig.CHECKPOINT_PATH)
            i = 0
            while self.dataloader.continue_flag:
                print("At Epoch: ", i, i/self.dataloader.nsamples, self.dataloader.nsamples)
                self.epoch.emit(int(100*i/self.dataloader.nsamples))
                i+=1
                images, coors_batch = self.dataloader.get_image_from_coor()
                pred, mid_features = sess.run([self.logits, self.end_points], feed_dict={self.images_test: images})
                if self.dataloader.write_to_folder_flag:
                    self.dataloader.save_predictions(pred, mid_features, coors_batch, self.wsi_list["Tumor"][self.wsi_iter])
        if self.dataloader.data_completed:
            print("Total time taken: ", time()-self.t0)
            # self.finished.emit()

    @pyqtSlot()
    def stop_call(self):
        """Stops feature generation and exits"""
        print("Stopping Testing..")
        self.dataloader.continue_flag = False
        self.dataloader.write_to_folder_flag = False
        # time.sleep(2)
        # if os.path.exists(LSTMDataConfig.RESULT_PATH):
        #     try:
        #         shutil.rmtree(LSTMDataConfig.RESULT_PATH)
        #         print("Result tree removed")
        #     except:
        #         pass
        self.epoch.emit(0)
        self.finished.emit()

class TestLSTMLabelSave(QObject):
    """Saves labels for selected patches in a separate directory"""
    finished = pyqtSignal()
    epoch = pyqtSignal(int)

    @pyqtSlot()
    def test(self):
        """Start saving labels for a list of WSIs"""
        self.wsi_iter = None
        self.wsi_list = os.listdir(LSTMDataConfig.RESULT_PATH)
        for i in range(len(self.wsi_list)):
            print(self.wsi_list[i])
            self.wsi_iter = i
            LSTMDataConfig.MASK_PATH = LSTMDataConfig.WSI_FOLDER_PATH + os.sep + "Tumor" + os.sep + "Mask_Tumor" + os.sep +\
                             self.wsi_list[self.wsi_iter] + '.tif'
            self.wsi_mask = ImageClass(LSTMDataConfig.MASK_PATH)
            self.test_once()
        print("Finished for all assigned WSIs")

    @pyqtSlot()
    def test_once(self):
        """Saves labels for a particular WSI"""
        # Saver and initialisation
        self.epoch.emit(0)
        wsi_name = self.wsi_list[self.wsi_iter]
        imlist = os.listdir(LSTMDataConfig.RESULT_PATH + os.sep + wsi_name)
        if not os.path.exists(LSTMDataConfig.LABEL_PATH + os.sep + wsi_name):
            os.mkdir(LSTMDataConfig.LABEL_PATH + os.sep + wsi_name)
        for i in range(len(imlist)):
            coors = imlist[i].split('(')[1].split(')')[0]
            w, h = list(map(int, coors.split(',')))
            im = np.array(self.wsi_mask.read_region((w, h), 0, (LSTMDataConfig.PATCH_SIZE*LSTMDataConfig.CONTEXT_DEPTH,
                                                       LSTMDataConfig.PATCH_SIZE * LSTMDataConfig.CONTEXT_DEPTH)).convert('1'))
            if np.mean(im) > 0:
                print("Found 1 with mean>0", i, len(imlist), self.wsi_iter)
            im = [[np.mean(im[i:i + LSTMDataConfig.PATCH_SIZE, j:j + LSTMDataConfig.PATCH_SIZE]) for i in
                range(0, LSTMDataConfig.PATCH_SIZE * LSTMDataConfig.CONTEXT_DEPTH,
                      LSTMDataConfig.PATCH_SIZE)] for j in
                range(0, LSTMDataConfig.PATCH_SIZE * LSTMDataConfig.CONTEXT_DEPTH,
                      LSTMDataConfig.PATCH_SIZE)]
            im = np.array(im).reshape(LSTMDataConfig.CONTEXT_DEPTH, LSTMDataConfig.CONTEXT_DEPTH, 1)
            # with open(LSTMDataConfig.LABEL_PATH + os.sep + wsi_name + os.sep + wsi_name + '_(' + str(w) + ', ' + str(h) + ')_label.pkl', 'wb') as fp:
            #     pickle.dump(im, fp)
            np.save(LSTMDataConfig.LABEL_PATH + os.sep + wsi_name + os.sep + wsi_name + '_(' + str(w) + ', ' + str(h) + ')_label.npy', im)

    @pyqtSlot()
    def stop_call(self):
        """Stop label saving process and exit"""
        print("Stopping Testing..")
        self.epoch.emit(0)
        self.finished.emit()