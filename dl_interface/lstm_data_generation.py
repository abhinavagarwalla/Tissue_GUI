import shutil
from time import time
import os
import random

import tensorflow as tf
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import numpy as np
import cv2
import glob
import pickle

from dl_interface.model_config import LSTMDataConfig
from nets import nets_factory
from interface.image_slide import ImageClass
from preprocessing import preprocessing_factory

from itertools import product
from scipy import ndimage

class DataMod():
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

    def get_image_from_coor(self):
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
                im = [[self.preprocessor.preprocess_image(im[i:i + LSTMDataConfig.PATCH_SIZE, j:j + LSTMDataConfig.PATCH_SIZE]) for i in
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
        # fc8 = mid_features['alexnet_v2/fc8']
        fc6 = mid_features['alexnet_v2/fc6']
        # fc7 = mid_features['alexnet_v2/fc7']
        preds = np.argmax(preds, axis=1)
        wsi_name = wsi_name.split(os.sep)[-1].split('.')[0]
        if not os.path.exists(LSTMDataConfig.RESULT_PATH + os.sep + wsi_name):
            os.mkdir(LSTMDataConfig.RESULT_PATH + os.sep + wsi_name)
        for i in range(len(coors_batch)):
            features = {'predictions': preds[i*LSTMDataConfig.CONTEXT_DEPTH**2: (i+1)*LSTMDataConfig.CONTEXT_DEPTH**2],
                        # 'fc8': fc8[i*LSTMDataConfig.CONTEXT_DEPTH**2: (i+1)*LSTMDataConfig.CONTEXT_DEPTH**2],
                        # 'fc7': fc7[i*LSTMDataConfig.CONTEXT_DEPTH**2: (i+1)*LSTMDataConfig.CONTEXT_DEPTH**2],
                        'fc6': fc6[i*LSTMDataConfig.CONTEXT_DEPTH**2: (i+1)*LSTMDataConfig.CONTEXT_DEPTH**2]}
            with open(LSTMDataConfig.RESULT_PATH + os.sep + wsi_name + os.sep + wsi_name + '_' + str(coors_batch[i]) + '_features.pkl', 'wb') as fp:
                pickle.dump(features, fp)

        # if self.write_to_folder_flag:
        #     np.save(LSTMDataConfig.RESULT_PATH + os.sep + wsi_name.split(os.sep)[-1].split('.')[0] + '_' + str(coors_batch[0]) + '_preds.npy', preds)
        #     np.save(LSTMDataConfig.RESULT_PATH + os.sep + wsi_name.split(os.sep)[-1].split('.')[0] + '_' + str(coors_batch[0]) + '_fc6.npy', fc6)
        #     np.save(LSTMDataConfig.RESULT_PATH + os.sep + wsi_name.split(os.sep)[-1].split('.')[0] + '_' + str(coors_batch[0]) + '_fc7.npy', fc7)
        #     np.save(LSTMDataConfig.RESULT_PATH + os.sep + wsi_name.split(os.sep)[-1].split('.')[0] + '_' + str(coors_batch[0]) + '_fc8.npy', fc8)

class TestLSTMSave(QObject):
    finished = pyqtSignal()
    epoch = pyqtSignal(int)

    def initialize(self):
        self.t0 = time()
        self.images_test = tf.placeholder(tf.float32, shape=(None, LSTMDataConfig.PATCH_SIZE, LSTMDataConfig.PATCH_SIZE, 3))
        self.logits, self.end_points = nets_factory.get_network_fn(name='alexnet', images=self.images_test,
                                                         num_classes=LSTMDataConfig.NUM_CLASSES, is_training=False)

    def init_data_loader(self):
        self.dataloader = DataMod(preprocessor='stain_norm')

        # if os.path.exists(LSTMDataConfig.RESULT_PATH):
        #     shutil.rmtree(Config.RESULT_PATH)
        # os.mkdir(Config.RESULT_PATH)
        # os.mkdir(Config.RESULT_PATH + os.sep + "predictions_png")

    @pyqtSlot()
    def test(self):
        self.wsi_iter = None
        self.classes = os.listdir(LSTMDataConfig.WSI_FOLDER_PATH)
        # self.wsi_list = dict((i, os.listdir(PatchConfig.WSI_FOLDER_PATH+os.sep+i)) for i in self.classes)
        self.wsi_list = dict((i, glob.glob(LSTMDataConfig.WSI_FOLDER_PATH + os.sep + i + os.sep + "*.tif")) for i in self.classes)
        self.classes_dict = dict((i, self.classes[i]) for i in range(len(self.classes)))
        self.initialize()
        # mlist = [i[:-5] for i in mlist]
        for i in range(30, 40):#len(self.wsi_list["Tumor"])):
            print(self.wsi_list["Tumor"][i])
            self.wsi_iter = i
            LSTMDataConfig.WSI_PATH = self.wsi_list["Tumor"][self.wsi_iter]
            LSTMDataConfig.MASK_PATH = LSTMDataConfig.WSI_FOLDER_PATH + os.sep + "Tumor" + os.sep + "Mask_Tissue" + os.sep +\
                             self.wsi_list["Tumor"][self.wsi_iter].split(os.sep)[-1]
            self.test_once()
        print("Finished for all assigned WSIs")

    @pyqtSlot()
    def test_once(self):
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
    finished = pyqtSignal()
    epoch = pyqtSignal(int)

    @pyqtSlot()
    def test(self):
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
        print("Stopping Testing..")
        self.epoch.emit(0)
        self.finished.emit()