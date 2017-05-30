from PIL import Image, ImageDraw
import tensorflow as tf
import model_definition
import cv2
from model_config import *
from scipy import ndimage
from itertools import product
import numpy as np
import openslide as ops
import math
from model_preprocess import Preprocess
from combine_predictions import combine
from time import  time

class Test():
    def __init__(self):
        self.t0 = time()
        self.wsi = ops.OpenSlide(WSI_PATH)
        self.images_test = tf.placeholder(tf.float32, shape=(None, PATCH_SIZE, PATCH_SIZE, 3))
        # Network
        self.net = model_definition.UNet()
        self.logits_test = self.net.inference(self.images_test)
        self.coors = self.get_coordinates()
        self.iter = 0
        self.nsamples = len(self.coors)
        self.nepoch = math.ceil(self.nsamples/BATCH_SIZE)
        self.preprocessor = Preprocess()

    def delete_inside(self, boxes):
        boxes = np.array(boxes)
        boxes_new = []
        for i in range(len(boxes)):
            a = boxes[(boxes[:,0]<boxes[i,0]) & (boxes[:,1]<boxes[i,1]) &
                      ((boxes[:,0]+boxes[:,2]) > (boxes[i,0]+boxes[i,2])) &
                      ((boxes[:,1]+boxes[:,3])>(boxes[i,1] + boxes[i,3]))]
            if len(a):
                print(len(a), a, boxes[i])
            else:
                boxes_new.append(boxes[i])
        return np.array(boxes_new)

    def get_coordinates(self):
        img = ndimage.imread(MASK_PATH)
        contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        a = np.array([cv2.contourArea(i) for i in contours[1]])
        b = np.array(contours[1])
        order = a.argsort()[::-1]
        a, b = a[order], b[order]
        threshArea, threshPoints = 200, 10
        boxes = [cv2.boundingRect(i) for i in b[a > threshArea] if len(i) > threshPoints]
        # shapely_box = [box(i[1], i[0], i[1]+i[3], i[0]+i[2]) for i in boxes]
        boxes = self.delete_inside(boxes)
        # print(boxes)

        boxes = boxes * pow(2, LEVEL_UPGRADE)
        print(boxes)
        coors = []
        ## Make it more complete
        for i in range(len(boxes)):
            a = range(max(0, boxes[i, 0]-DIFF_SIZE),
                      min(self.wsi.level_dimensions[LEVEL_FETCH][0],
                          boxes[i, 0] + boxes[i, 2] + DIFF_SIZE), OUTPUT_SIZE)
            b = range(max(0, boxes[i, 1]-DIFF_SIZE),
                      min(self.wsi.level_dimensions[LEVEL_FETCH][1],
                          boxes[i, 1] + boxes[i, 3] + DIFF_SIZE), OUTPUT_SIZE)
            coors.extend(list(product(a, b)))
        return coors

    def get_image_from_coor(self):
        image_batch = []
        coor_batch = []
        while len(coor_batch)!=BATCH_SIZE:
            # re = random.randint(0, self.nsamples)
            im = np.array(self.wsi.read_region((pow(2, LEVEL_FETCH)*self.coors[self.iter][0],
                                                pow(2, LEVEL_FETCH)*self.coors[self.iter][1]),
                                               LEVEL_FETCH,
                                               (PATCH_SIZE, PATCH_SIZE)).convert('RGB'))
            if np.mean(im)<= 240:
                im = self.preprocessor.stain_normalisation(im)
                image_batch.append(im[:,:,::-1]) #RGB to BGR
                coor_batch.append(self.coors[self.iter])
            self.iter += 1
            if self.iter==self.nsamples:
                break

        image_batch = np.array(image_batch).reshape(-1, PATCH_SIZE, PATCH_SIZE, 3)
        return image_batch, coor_batch

    def save_predictions(self, preds, coors_batch, images):
        preds = (np.array(preds)*100).astype(np.uint8)
        for i in range(BATCH_SIZE):
            # im_tumor = Image.fromarray(preds[i, :, :, 0])
            # im_non_tumor = Image.fromarray(preds[i, :, :, 1])
            # im_tumor.save('results\\' + str(coors_batch[i]) + "_tumor.png")
            # im_non_tumor.save('results\\' + str(coors_batch[i]) + "_non_tumor.png")
            cv2.imwrite("results\\" + str(coors_batch[i]) + "_tumor.png", preds[i, :, :, 0])
            cv2.imwrite("results\\" + str(coors_batch[i]) + "_non_tumor.png", preds[i, :, :, 1])
            # orim = Image.fromarray(images[i])
            # orim.save('results\\' + str(coors_batch[i]) + "_orig.png")

    def test(self):
        # Saver and initialisation
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, CHECKPOINT_PATH)
            for i in range(self.nepoch):
                print("At Epoch: ", i)
                images, coors_batch = self.get_image_from_coor()
                pred = sess.run(self.logits_test, feed_dict={self.images_test: images})
                self.save_predictions(pred, coors_batch, images)
            print("Done.")
        combine()
        print("Total time taken: ", time()-self.t0)