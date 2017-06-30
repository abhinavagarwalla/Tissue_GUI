# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import shutil
from time import time

import tensorflow as tf
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from dl_interface.combine_predictions import combine
from dl_interface.data_io import Data
from dl_interface.model_config import *
from nets import nets_factory


class Test(QObject):
    finished = pyqtSignal()
    epoch = pyqtSignal(int)

    def initialize(self):
        self.t0 = time()
        self.images_test = tf.placeholder(tf.float32, shape=(None, Config.PATCH_SIZE, Config.PATCH_SIZE, 3))
        self.output = nets_factory.get_network_fn(name='unet', images=self.images_test, is_training=False)
        self.dataloader = Data(preprocessor='stain_norm', outshape=self.output.shape)

        if os.path.exists(Config.RESULT_PATH):
            shutil.rmtree(Config.RESULT_PATH)
        os.mkdir(Config.RESULT_PATH)
        os.mkdir(Config.RESULT_PATH + os.sep + "predictions_png")

    @pyqtSlot()
    def test(self):
        # Saver and initialisation
        self.initialize()
        saver = tf.train.Saver()
        self.epoch.emit(0)
        with tf.Session() as sess:
            saver.restore(sess, Config.CHECKPOINT_PATH)
            i = 0
            while self.dataloader.continue_flag:
                print("At Epoch: ", i, (i*Config.BATCH_SIZE)/self.dataloader.nsamples, self.dataloader.nsamples)
                self.epoch.emit(int((100*i*Config.BATCH_SIZE)/self.dataloader.nsamples))
                i+=1
                images, coors_batch = self.dataloader.get_image_from_coor()
                if len(images)==Config.BATCH_SIZE:
                    pred = sess.run(self.output, feed_dict={self.images_test: images})
                    if self.dataloader.write_to_folder_flag:
                        self.dataloader.save_predictions(pred, coors_batch)
        if self.dataloader.data_completed:
            combine()
            print("Total time taken: ", time()-self.t0)
            self.finished.emit()

    @pyqtSlot()
    def stop_call(self):
        print("Stopping Testing..")
        self.dataloader.continue_flag = False
        self.dataloader.write_to_folder_flag = False
        # time.sleep(2)
        if os.path.exists(Config.RESULT_PATH):
            try:
                shutil.rmtree(Config.RESULT_PATH)
                print("Result tree removed")
            except:
                pass
        self.epoch.emit(0)
        self.finished.emit()