# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains utilities for batch generation for 2D-LSTMs"""

import scipy.io as sio
import os
import pickle
import numpy as np
import random
import glob

from dl_interface.model_config import LSTMTrainConfig, LSTMValidConfig
from dl_interface.model_config import LSTMDataConfig
from interface.image_slide import ImageClass

import cv2
from preprocessing import preprocessing_factory
from itertools import product
from scipy import ndimage
import scipy.io as sio

class LSTMTrainMatDataIter():
    """Generates batches from file names specified in mat file
    """
    def __init__(self):
        self.iter = 0
        plist = sio.loadmat('resource/wsi_names_list.mat')['wsi_names']
        self.images_list = [LSTMTrainConfig.DATA_IMAGES_PATH + os.sep + i[:9] + os.sep + i for i in plist]
        self.num_samples = len(self.images_list)

    def next_batch(self):
        """Fetches the next batch for training
        Returns:
            x: 2D-Grid of size N x M x D
            y: Labels of size N x M
            cnn_logits: Logits from pretrained CNN, size: N x M x num_classes
            cnn_y: Labels from pretrained CNN, size: N x M
        """
        x = []
        y = []
        cnn_logits = []
        cnn_y = []
        for i in range(LSTMTrainConfig.batch_size):
            feat = pickle.load(open(self.images_list[self.iter], 'rb'))
            wsi_name = self.images_list[self.iter].split(os.sep)[-1]
            label = np.load(LSTMTrainConfig.DATA_LABELS_PATH + os.sep + wsi_name.split('_(')[0] +
                            os.sep + wsi_name.replace('features.pkl', 'label.npy'))
            x.append(feat['fc6'].reshape(LSTMTrainConfig.PATCH_SIZE, LSTMTrainConfig.PATCH_SIZE, LSTMTrainConfig.CHANNELS))
            y.append(label)
            cnn_y.append(feat['predictions'].reshape(LSTMTrainConfig.PATCH_SIZE, LSTMTrainConfig.PATCH_SIZE, 1))
            cnn_logits.append(feat['fc8'].reshape(LSTMTrainConfig.PATCH_SIZE, LSTMTrainConfig.PATCH_SIZE, LSTMTrainConfig.NUM_CLASSES))
            self.iter += 1
            if self.iter == self.num_samples:
                self.iter = 0
        return x, y, cnn_y, cnn_logits

class LSTMTrainPNDataIter():
    """Generates batches from two mat files, one containing positives and other negatives
    """
    def __init__(self):
        self.iter = 0
        self.tumor_list = sio.loadmat('resource/tumor_wsi_list.mat')['wsi_names']
        self.non_tumor_list = sio.loadmat('resource/non_tumor_wsi_list.mat')['wsi_names']
        random.shuffle(self.non_tumor_list)
        random.shuffle(self.tumor_list)
        self.images_list = [LSTMTrainConfig.DATA_IMAGES_PATH + os.sep + i[:9] + os.sep + i for i in self.tumor_list]
        self.images_list.extend([LSTMTrainConfig.DATA_IMAGES_PATH + os.sep + i[:9] + os.sep + i for i in self.non_tumor_list[:int(1.5*len(self.tumor_list))]])
        random.shuffle(self.images_list)
        self.num_samples = len(self.images_list)

    def next_batch(self):
        """Fetches the next batch for training
        Returns:
            x: 2D-Grid of size N x M x D
            y: Labels of size N x M
            cnn_logits: Logits from pretrained CNN, size: N x M x num_classes
            cnn_y: Labels from pretrained CNN, size: N x M
        """
        x = []
        y = []
        cnn_logits = []
        cnn_y = []
        for i in range(LSTMTrainConfig.batch_size):
            feat = pickle.load(open(self.images_list[self.iter], 'rb'))
            wsi_name = self.images_list[self.iter].split(os.sep)[-1]
            label = np.load(LSTMTrainConfig.DATA_LABELS_PATH + os.sep + wsi_name.split('_(')[0] +
                            os.sep + wsi_name.replace('features.pkl', 'label.npy'))
            x.append(feat['fc6'].reshape(LSTMTrainConfig.PATCH_SIZE, LSTMTrainConfig.PATCH_SIZE, LSTMTrainConfig.CHANNELS))
            y.append(label)
            cnn_y.append(feat['predictions'].reshape(LSTMTrainConfig.PATCH_SIZE, LSTMTrainConfig.PATCH_SIZE, 1))
            cnn_logits.append(feat['fc8'].reshape(LSTMTrainConfig.PATCH_SIZE, LSTMTrainConfig.PATCH_SIZE, LSTMTrainConfig.NUM_CLASSES))
            self.iter += 1
            if self.iter == self.num_samples:
                self.iter = 0
        return x, y, cnn_y, cnn_logits

class LSTMValidDataIter():
    """Batch Generation for processing all of validation data
    """
    def __init__(self):
        self.wsi_list = glob.glob(LSTMValidConfig.DATA_IMAGES_PATH + os.sep + 'Tumor*')
        self.images_list = []
        for i in self.wsi_list:
            self.images_list.extend(glob.glob(i + os.sep + '*'))
        random.shuffle(self.images_list)
        self.num_samples = len(self.images_list)
        self.iter = 0
        self.save_coor = None

    def next_batch(self):
        """Fetches the next batch for validation
        Returns:
            x: 2D-Grid of size N x M x D
            y: Labels of size N x M
            cnn_logits: Logits from pretrained CNN, size: N x M x num_classes
            cnn_y: Labels from pretrained CNN, size: N x M
        """
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
            self.save_coor = self.images_list[self.iter].replace('_features.pkl', '_preds.npy')
            self.names.append(self.images_list[self.iter].replace('_features.pkl', '_preds.npy').split(os.sep)[-1])
            self.iter += 1
            if self.iter == self.num_samples:
                self.iter = 0
        return x, y, cnn_y, cnn_logits

    def save_predictions(self, probs):
        """Saves output predictions for the current batch as numpy array
        Args:
            probs: Class probabilites of size batch_size x num_classes    
        """
        for i in range(len(self.names)):
            start = i*LSTMValidConfig.PATCH_SIZE*LSTMValidConfig.PATCH_SIZE
            end = (i+1)*LSTMValidConfig.PATCH_SIZE*LSTMValidConfig.PATCH_SIZE
            np.save(LSTMValidConfig.log_dir + os.sep + 'predictions' + os.sep + self.names[i], probs[start:end,])

    def combine_prediction(self):
        """Dummy function, edit if sophisticated combination required
        """
        pred_list = glob.glob(LSTMValidConfig.DATA_IMAGES_PATH + os.sep + 'predictions' + os.sep + '*')
        print(pred_list)

class LSTMDataMod():
    """Fetches batches for generating features for generating 2D-LSTM data"""
    def __init__(self, preprocessor):
        self.wsi = ImageClass(LSTMDataConfig.WSI_PATH)
        self.preprocessor = preprocessing_factory.get_preprocessing_fn(name=preprocessor)
        self.coors = self.get_coordinates()
        self.nsamples = len(self.coors)
        self.iter = 0
        self.continue_flag = True
        self.data_completed = False
        self.write_to_folder_flag = True

    def delete_inside(self, boxes):
        """Delete coordinate box inside another box"""
        boxes = np.array(boxes)
        boxes_new = []
        for i in range(len(boxes)):
            a = boxes[(boxes[:, 0] < boxes[i, 0]) & (boxes[:, 1] < boxes[i, 1]) &
                      ((boxes[:, 0] + boxes[:, 2]) > (boxes[i, 0] + boxes[i, 2])) &
                      ((boxes[:, 1] + boxes[:, 3]) > (boxes[i, 1] + boxes[i, 3]))]
            if len(a):
                print(len(a), a, boxes[i])
            else:
                if (boxes[i, 0]+boxes[i,2])< 0.05*self.wsi.level_dimensions[LSTMDataConfig.LEVEL_FETCH+LSTMDataConfig.LEVEL_UPGRADE][0]:
                    continue
                if boxes[i, 0] > 0.95 * self.wsi.level_dimensions[LSTMDataConfig.LEVEL_FETCH+LSTMDataConfig.LEVEL_UPGRADE][0]:
                    continue
                if (boxes[i, 1]+boxes[i,3])< 0.05*self.wsi.level_dimensions[LSTMDataConfig.LEVEL_FETCH+LSTMDataConfig.LEVEL_UPGRADE][1]:
                    continue
                if boxes[i, 1] > 0.95 * self.wsi.level_dimensions[LSTMDataConfig.LEVEL_FETCH+LSTMDataConfig.LEVEL_UPGRADE][1]:
                    continue
                boxes_new.append(boxes[i])
        return np.array(boxes_new)

    def get_coordinates(self):
        """Get coordinates for all patches, either from Tissue Mask or Label Directory"""
        if not LSTMDataConfig.READ_FROM_COOR:
            img = ndimage.imread(LSTMDataConfig.MASK_PATH)
            contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            a = np.array([cv2.contourArea(i) for i in contours[1]])
            b = np.array(contours[1])
            order = a.argsort()[::-1]
            a, b = a[order], b[order]
            threshArea, threshPoints = 200, 10
            boxes = [cv2.boundingRect(i) for i in b[a > threshArea] if len(i) > threshPoints]
            boxes = self.delete_inside(boxes)

            boxes = boxes * pow(2, LSTMDataConfig.LEVEL_UPGRADE)
            print(boxes)
            coors = []
            for i in range(len(boxes)):
                a = range(max(0, boxes[i, 0]),
                          min(self.wsi.level_dimensions[LSTMDataConfig.LEVEL_FETCH][0],
                              boxes[i, 0] + boxes[i, 2]), int((1-LSTMDataConfig.STRIDE)*LSTMDataConfig.CONTEXT_DEPTH*LSTMDataConfig.PATCH_SIZE))
                b = range(max(0, boxes[i, 1]),
                          min(self.wsi.level_dimensions[LSTMDataConfig.LEVEL_FETCH][1],
                              boxes[i, 1] + boxes[i, 3]), int((1-LSTMDataConfig.STRIDE)*LSTMDataConfig.CONTEXT_DEPTH*LSTMDataConfig.PATCH_SIZE))
                coors.extend(list(product(a, b)))
            random.shuffle(coors)
            print("All coordinates has been fetched")
            return coors
        else:
            coors = []
            wsp = LSTMDataConfig.LABEL_PATH + os.sep + LSTMDataConfig.WSI_PATH.split(os.sep)[-1][:-4]
            for j in os.listdir(wsp):
                coorst = j.split('(')[1].split(')')[0]
                w, h = list(map(int, coorst.split(',')))
                coors.append((w, h))
            print("All coordinates have been fetched")
            return coors

    def get_image_from_coor(self):
        """Fetch and normalize image region for the coordinates"""
        # assert (LSTMDataConfig.BATCH_SIZE == 1)
        image_batch = []
        coor_batch = []
        label_batch = []
        while len(coor_batch) != int(LSTMDataConfig.BATCH_SIZE/(LSTMDataConfig.CONTEXT_DEPTH**2)):
            # re = random.randint(0, self.nsamples)
            im = np.array(self.wsi.read_region((pow(2, LSTMDataConfig.LEVEL_FETCH) * self.coors[self.iter][0],
                                                pow(2, LSTMDataConfig.LEVEL_FETCH) * self.coors[self.iter][1]),
                                               LSTMDataConfig.LEVEL_FETCH,
                                               (LSTMDataConfig.CONTEXT_DEPTH*LSTMDataConfig.PATCH_SIZE,
                                                LSTMDataConfig.CONTEXT_DEPTH*LSTMDataConfig.PATCH_SIZE)).convert('RGB'))
            # Modification here.
            if np.mean(im) <= 220:
                im = np.array(im).reshape(LSTMDataConfig.CONTEXT_DEPTH * LSTMDataConfig.PATCH_SIZE,
                                                            LSTMDataConfig.CONTEXT_DEPTH * LSTMDataConfig.PATCH_SIZE, 3)
                im = im/127.5 - 1
                # im = [[self.preprocessor.preprocess_image(im[i:i + LSTMDataConfig.PATCH_SIZE, j:j + LSTMDataConfig.PATCH_SIZE]) for i in
                #                 range(0, LSTMDataConfig.PATCH_SIZE * LSTMDataConfig.CONTEXT_DEPTH,
                #                       LSTMDataConfig.PATCH_SIZE)] for j in
                #                range(0, LSTMDataConfig.PATCH_SIZE * LSTMDataConfig.CONTEXT_DEPTH,
                #                      LSTMDataConfig.PATCH_SIZE)]
                im = [[im[i:i + LSTMDataConfig.PATCH_SIZE, j:j + LSTMDataConfig.PATCH_SIZE] for i in
                       range(0, LSTMDataConfig.PATCH_SIZE * LSTMDataConfig.CONTEXT_DEPTH,
                             LSTMDataConfig.PATCH_SIZE)] for j in
                      range(0, LSTMDataConfig.PATCH_SIZE * LSTMDataConfig.CONTEXT_DEPTH,
                            LSTMDataConfig.PATCH_SIZE)]
                im = np.array(im).reshape(-1, LSTMDataConfig.PATCH_SIZE, LSTMDataConfig.PATCH_SIZE, 3)

                for i in im:
                    image_batch.append(i[:, :, ::-1])  # RGB to BGR
                coor_batch.append(self.coors[self.iter])
            self.iter += 1
            if self.iter == self.nsamples:
                self.continue_flag = False
                self.data_completed = True
                break

        print("Completed iterations: ", self.iter, "/", self.nsamples)
        # image_batch = np.array(image_batch).reshape(-1, LSTMDataConfig.PATCH_SIZE, LSTMDataConfig.PATCH_SIZE, 3)
        return image_batch, coor_batch

    def save_predictions(self, preds, mid_features, coors_batch, wsi_name, images=None):
        """Save features as encoded by CNN
        Args:
            preds: Final predictions from CNN
            mid_features: All intermediate end-points like fc6, fc7, etc.
            coors_batch: Coordinates at WSI level for preds
            wsi_name: WSI name for preds
            images: Additional image to be saved, if any
        """
        fc8 = mid_features['alexnet_v2/fc8']
        fc6 = mid_features['alexnet_v2/fc6']
        # fc7 = mid_features['alexnet_v2/fc7']
        preds = np.argmax(preds, axis=1)
        wsi_name = wsi_name.split(os.sep)[-1].split('.')[0]
        if not os.path.exists(LSTMDataConfig.RESULT_PATH + os.sep + wsi_name):
            os.mkdir(LSTMDataConfig.RESULT_PATH + os.sep + wsi_name)
        for i in range(len(coors_batch)):
            features = {'predictions': preds[i*LSTMDataConfig.CONTEXT_DEPTH**2: (i+1)*LSTMDataConfig.CONTEXT_DEPTH**2],
                        'fc8': fc8[i*LSTMDataConfig.CONTEXT_DEPTH**2: (i+1)*LSTMDataConfig.CONTEXT_DEPTH**2],
                        # 'fc7': fc7[i*LSTMDataConfig.CONTEXT_DEPTH**2: (i+1)*LSTMDataConfig.CONTEXT_DEPTH**2],
                        'fc6': fc6[i*LSTMDataConfig.CONTEXT_DEPTH**2: (i+1)*LSTMDataConfig.CONTEXT_DEPTH**2]}
            with open(LSTMDataConfig.RESULT_PATH + os.sep + wsi_name + os.sep + wsi_name + '_' + str(coors_batch[i]) + '_features.pkl', 'wb') as fp:
                pickle.dump(features, fp)

        # conv5 = mid_features['alexnet_v2/conv5']
        # wsi_name = wsi_name.split(os.sep)[-1].split('.')[0]
        # if not os.path.exists(LSTMDataConfig.RESULT_PATH + os.sep + wsi_name):
        #     os.mkdir(LSTMDataConfig.RESULT_PATH + os.sep + wsi_name)
        # for i in range(len(coors_batch)):
        #     np.save(LSTMDataConfig.RESULT_PATH + os.sep + wsi_name + os.sep + wsi_name + '_' + str(coors_batch[i]) + '_conv5.npy',
        #             conv5[i*LSTMDataConfig.CONTEXT_DEPTH**2: (i+1)*LSTMDataConfig.CONTEXT_DEPTH**2])

        # if self.write_to_folder_flag:
        #     np.save(LSTMDataConfig.RESULT_PATH + os.sep + wsi_name.split(os.sep)[-1].split('.')[0] + '_' + str(coors_batch[0]) + '_preds.npy', preds)
        #     np.save(LSTMDataConfig.RESULT_PATH + os.sep + wsi_name.split(os.sep)[-1].split('.')[0] + '_' + str(coors_batch[0]) + '_fc6.npy', fc6)
        #     np.save(LSTMDataConfig.RESULT_PATH + os.sep + wsi_name.split(os.sep)[-1].split('.')[0] + '_' + str(coors_batch[0]) + '_fc7.npy', fc7)
        #     np.save(LSTMDataConfig.RESULT_PATH + os.sep + wsi_name.split(os.sep)[-1].split('.')[0] + '_' + str(coors_batch[0]) + '_fc8.npy', fc8)
