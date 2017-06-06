import os
import shutil
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from itertools import product
import time
import cv2
import numpy as np
from scipy import ndimage
import random
from dl_interface.model_config import PatchConfig
from interface.image_slide import ImageClass
from PIL import Image
import glob

class PatchGenerator(QObject):
    finished = pyqtSignal()

    def initialize(self):
        # self.wsi = ImageClass(PatchConfig.WSI_PATH)
        # self.coors = self.get_coordinates()
        # self.nsamples = len(self.coors)
        self.iter = 0
        self.class_iter = 0
        self.wsi_iter = 0

        self.continue_flag = True
        self.data_completed = False
        self.write_to_folder_flag = True
        self.wsi_completed = False

        self.classes = os.listdir(PatchConfig.WSI_FOLDER_PATH)
        # self.wsi_list = dict((i, os.listdir(PatchConfig.WSI_FOLDER_PATH+os.sep+i)) for i in self.classes)
        self.wsi_list = dict((i, glob.glob(PatchConfig.WSI_FOLDER_PATH + os.sep + i + os.sep + "*.tif")) for i in self.classes)
        self.classes_dict = dict((i, self.classes[i]) for i in range(len(self.classes)))

        if os.path.exists(PatchConfig.RESULT_PATH):
            shutil.rmtree(PatchConfig.RESULT_PATH)
        os.mkdir(PatchConfig.RESULT_PATH)
        os.mkdir(PatchConfig.RESULT_PATH + os.sep + "Coors")
        os.mkdir(PatchConfig.RESULT_PATH + os.sep + "Ambiguous")
        [os.mkdir(PatchConfig.RESULT_PATH + os.sep + i) for i in self.classes]

    def run(self):
        self.initialize()
        while(self.continue_flag):
            time.sleep(3)
            print(PatchConfig.WSI_FOLDER_PATH, self.classes)
            for i in range(len(self.wsi_list["Tumor"])):
                self.wsi_iter = i
                self.wsi = ImageClass(self.wsi_list["Tumor"][self.wsi_iter])
                self.tumor_wsi = ImageClass(PatchConfig.WSI_FOLDER_PATH + os.sep + "Tumor" + os.sep + "Mask_Tumor" +\
                                            os.sep + self.wsi_list["Tumor"][self.wsi_iter].split(os.sep)[-1])
                self.coors = self.get_coordinates()
                self.coor_labels = []
                self.nsamples = len(self.coors)
                while(not self.wsi_completed):
                    if self.continue_flag:
                        self.get_and_save_batch()
                    else:
                        break
                self.wsi_completed = False
                np.save(PatchConfig.WSI_FOLDER_PATH + os.sep + "Coors" + os.sep + \
                        self.wsi_list["Tumor"][self.wsi_iter].split('.')[0] + '_coors.npy', np.array(self.coors))
                np.save(PatchConfig.WSI_FOLDER_PATH + os.sep + "Coors" + os.sep + \
                        self.wsi_list["Tumor"][self.wsi_iter].split('.')[0] + '_coors_labels.npy', np.array(self.coor_labels))
                print("WSI has been completed, iterating to next")
            self.data_completed = True

    def get_and_save_batch(self):
        image_batch = []
        coor_batch = []
        folder_list = []
        while len(coor_batch) != PatchConfig.BATCH_SIZE:
            # re = random.randint(0, self.nsamples)
            im = np.array(self.wsi.read_region((pow(2, PatchConfig.LEVEL_FETCH) * self.coors[self.iter][0],
                                                pow(2, PatchConfig.LEVEL_FETCH) * self.coors[self.iter][1]),
                                               PatchConfig.LEVEL_FETCH,
                                               (PatchConfig.PATCH_SIZE, PatchConfig.PATCH_SIZE)).convert('RGB'))
            tover = np.mean(np.array(self.tumor_wsi.read_region((pow(2, PatchConfig.LEVEL_FETCH) * self.coors[self.iter][0],
                                                pow(2, PatchConfig.LEVEL_FETCH) * self.coors[self.iter][1]), PatchConfig.LEVEL_FETCH,
                                                                (PatchConfig.PATCH_SIZE,
                                                                 PatchConfig.PATCH_SIZE)).convert('1')))

            if tover > 0.60:
                image_batch.append(im[:, :, ::-1])  # RGB to BGR
                coor_batch.append(self.coors[self.iter])
                folder_list.append("Tumor")
            elif tover < 0.25:
                if np.mean(im) <= 180:
                    prob = random.random()
                    if prob < 0.10:
                        image_batch.append(im[:, :, ::-1])  # RGB to BGR
                        coor_batch.append(self.coors[self.iter])
                elif np.mean(im) >= 240:
                    pass
                else:
                    prob = random.random()
                    if prob < 0.001:
                        image_batch.append(im[:, :, ::-1])  # RGB to BGR
                        coor_batch.append(self.coors[self.iter])
                folder_list.append("Normal")
            elif abs(tover-0.50)<0.05:
                image_batch.append(im[:, :, ::-1])  # RGB to BGR
                coor_batch.append(self.coors[self.iter])
                folder_list.append("Ambiguous")

            self.coor_labels.append(tover)
            self.iter += 1
            if self.iter == self.nsamples:
                self.wsi_completed = True
                break

        image_batch = np.array(image_batch).reshape(-1, PatchConfig.PATCH_SIZE, PatchConfig.PATCH_SIZE, 3)
        self.save_predictions(image_batch, coor_batch, folder_list)

    def save_predictions(self, images, coors_batch, folder_name):
        print("Saving Predictions..", self.wsi_iter, self.iter)
        for i in range(len(images)):
            if self.write_to_folder_flag:
                orim = Image.fromarray(images[i])
                orim.save(PatchConfig.RESULT_PATH + os.sep + folder_name[i] + os.sep + \
                          'level_' + str(PatchConfig.LEVEL_FETCH) + '_' + str(coors_batch[i]) + "_" + \
                              self.wsi_list["Tumor"][self.wsi_iter].split(os.sep)[-1].split('.')[0] + ".png")


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
                boxes_new.append(boxes[i])
        return np.array(boxes_new)

    def get_coordinates(self):
        img = ndimage.imread(PatchConfig.WSI_FOLDER_PATH + os.sep + "Tumor" + os.sep + "Mask_Tissue" + os.sep +\
                             self.wsi_list["Tumor"][self.wsi_iter].split(os.sep)[-1])
        contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        a = np.array([cv2.contourArea(i) for i in contours[1]])
        b = np.array(contours[1])
        order = a.argsort()[::-1]
        a, b = a[order], b[order]
        threshArea, threshPoints = 200, 10
        boxes = [cv2.boundingRect(i) for i in b[a > threshArea] if len(i) > threshPoints]
        boxes = self.delete_inside(boxes)

        boxes = boxes * pow(2, PatchConfig.LEVEL_UPGRADE)
        print(boxes)
        coors = []
        for i in range(1):
            a = range(max(0, boxes[i, 0]),
                      min(self.wsi.level_dimensions[PatchConfig.LEVEL_FETCH][0],
                          boxes[i, 0] + boxes[i, 2]), PatchConfig.PATCH_SIZE)
            b = range(max(0, boxes[i, 1]),
                      min(self.wsi.level_dimensions[PatchConfig.LEVEL_FETCH][1],
                          boxes[i, 1] + boxes[i, 3]), PatchConfig.PATCH_SIZE)
            coors.extend(list(product(a, b)))
        return coors

    @pyqtSlot()
    def stop_call(self):
        self.continue_flag = False
        self.finished.emit()