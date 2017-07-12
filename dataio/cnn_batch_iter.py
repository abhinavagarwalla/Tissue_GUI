# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains utilities for batch generation for CNNs"""

import scipy.io as sio
import os
import pickle
import numpy as np
from interface.image_slide import ImageClass
from preprocessing import preprocessing_factory

from dl_interface.model_config import CNN2TrainConfig

class CNNDataIter():
    def __init__(self):
        plist = sio.loadmat('resource/wsi_names_list.mat')['wsi_names']
        self.images_list = [CNN2TrainConfig.DATA_IMAGES_PATH + os.sep + i[:9] + os.sep + i for i in plist]
        self.num_samples = len(self.images_list)
        self.iter = 0
        self.preprocessor = preprocessing_factory.get_preprocessing_fn(name='camelyon')

    def next_batch(self):
        """Fetches the next batch for training
        Returns:
            x: 4D vector of training images, [batch_size, patch_size, patch_size, 3]
            y: 2D vector of labels, [batch_size, num_class]
        """
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
                    img = im[j:j + CNN2TrainConfig.IMAGE_SIZE, i:i + CNN2TrainConfig.IMAGE_SIZE]/255 # - 1.0
                    x.append(img)

            label = np.load(CNN2TrainConfig.DATA_LABELS_PATH + os.sep + wsi_id +
                            os.sep + patch_id.split(os.sep)[-1].replace('features.pkl', 'label.npy')).reshape(-1)
            label_inv = 1-label
            label = np.vstack((label_inv, label)).T
            y.extend(label)
            self.iter += 1
            if self.iter == self.num_samples:
                self.iter = 0
        return x, y