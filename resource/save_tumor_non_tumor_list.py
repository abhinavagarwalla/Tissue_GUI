import scipy.io as sio
import random
import os
import pickle
import numpy as np
import glob
from pathos.multiprocessing import ProcessPool as Pool

class LSTMTrainConfig():
    DATA_IMAGES_PATH = 'F:\\abhinav\\patches\\lstm_data\\train'
    DATA_LABELS_PATH = 'F:\\abhinav\\patches\\lstm_data_label'
    PATCH_SIZE = 8
    CHANNELS = 4096
    HIDDEN_SIZE = 512
    NUM_CLASSES = 2
    log_dir = 'F:\\abhinav\\patches'
    batch_size = 10
    num_epochs = 10 #None
    checkpoint_file = None #'F:\\abhinav\\patches\\log_lstm_run2\\model.ckpt-29511'
    initial_learning_rate = 0.0001
    learning_rate_decay_factor = 0.5
    num_epochs_before_decay = 2

class DataIter():
    def __init__(self):
        self.iter = 0
        self.wsi_list = glob.glob(LSTMTrainConfig.DATA_IMAGES_PATH + os.sep + '*')
        self.images_list = []
        for i in self.wsi_list:
            self.images_list.extend(glob.glob(i + os.sep + '*'))
        random.shuffle(self.images_list)
        self.num_samples = len(self.images_list)
        self.tumour_wsi_list = []
        self.normal_wsi_list = []
        self.p = Pool(16)

    def add(self, i):
        wsi_name = self.images_list[i].split(os.sep)[-1]
        label = np.load(LSTMTrainConfig.DATA_LABELS_PATH + os.sep + wsi_name.split('_(')[0] +
                        os.sep + wsi_name.replace('features.pkl', 'label.npy'))
        label = np.reshape(label, [-1])
        tp = np.sum(label >= 0.5)
        if (tp > 0):
            self.tumour_wsi_list.append(wsi_name)
        else:
            self.normal_wsi_list.append(wsi_name)
        print(i)

    def save_tumor_non_tumor(self):
        # self.p.map(self.add, range(self.num_samples))
        for i in range(self.num_samples):
            self.add(i)
        sio.savemat(os.path.join(LSTMTrainConfig.log_dir, 'tumor_wsi_list'),
                    {'wsi_names': self.tumour_wsi_list})
        sio.savemat(os.path.join(LSTMTrainConfig.log_dir, 'non_tumor_wsi_list'),
                    {'wsi_names': self.normal_wsi_list})
if __name__=="__main__":
    d = DataIter()
    d.save_tumor_non_tumor()