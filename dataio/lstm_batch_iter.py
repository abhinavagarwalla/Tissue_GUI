import scipy.io as sio
import os
import pickle
import numpy as np
import random
import glob

from dl_interface.model_config import LSTMTrainConfig

class LSTMTrainMatDataIter():
    def __init__(self):
        self.iter = 0
        plist = sio.loadmat('resource/wsi_names_list.mat')['wsi_names']
        self.images_list = [LSTMTrainConfig.DATA_IMAGES_PATH + os.sep + i[:9] + os.sep + i for i in plist]
        self.num_samples = len(self.images_list)

    def next_batch(self):
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
