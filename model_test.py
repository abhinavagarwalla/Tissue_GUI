from PIL import Image
import tensorflow as tf
import model_definition
import cv2
from model_config import *
from scipy import ndimage
from itertools import product
import numpy as np
import openslide as ops

def delete_inside(boxes):
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

def get_coordinates():
    img = ndimage.imread(MASK_PATH)
    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    a = np.array([cv2.contourArea(i) for i in contours[1]])
    b = np.array(contours[1])
    threshArea, threshPoints = 200, 10
    boxes = [cv2.boundingRect(i) for i in b[a > threshArea] if len(i) > threshPoints]
    # shapely_box = [box(i[1], i[0], i[1]+i[3], i[0]+i[2]) for i in boxes]
    boxes = delete_inside(boxes)
    print(boxes)

    boxes = boxes * pow(2, LEVEL_UPGRADE)
    coors = []
    for i in range(len(boxes)):
        a = range(boxes[i, 0], boxes[i, 0] + boxes[i, 2], PATCH_SIZE)
        b = range(boxes[i, 1], boxes[i, 1] + boxes[i, 3], PATCH_SIZE)
        coors.extend(list(product(a, b)))
    return coors

def get_image_from_coor(wsi, coors):
    image_batch = []
    for i in range(BATCH_SIZE):
        im = np.array(wsi.read_region((coors[0][0], coors[0][1]), LEVEL_FETCH, (PATCH_SIZE, PATCH_SIZE)).convert('RGB'))
        image_batch.append(im)

    ## Also, convert RGB to BGR
    image_batch = np.array(image_batch).reshape(-1, PATCH_SIZE, PATCH_SIZE, 3)
    return image_batch

def test():
    coors = get_coordinates()

    wsi = ops.OpenSlide(WSI_PATH)
    images_test = tf.placeholder(tf.float32, shape=(BATCH_SIZE, PATCH_SIZE, PATCH_SIZE, 3))
    # Network
    net = model_definition.UNet()
    logits_test = net.inference(images_test)
    # Saver and initialisation
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        saver.restore(sess, CHECKPOINT_PATH)
        images = get_image_from_coor(wsi, coors)
        pred = sess.run(logits_test, feed_dict={images_test: images})
        print(pred.shape, pred[0])
    # with tf.Session() as sess:
    #     # Initialise and load variables
    #     sess.run(init)
    #     saver.restore(sess, model_path)
    #     result = batch_processing(image, mask, sess, logits_test, images_test)
    # duration = time.time() - start_time
    # print('Processing Time: %.2f minutes/tile' % (duration / 60))
    # return result