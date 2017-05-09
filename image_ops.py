import os
import cv2 as cv
import numpy as np
import openslide as ops

import sys
import math
# import Image
import scipy.io as sio
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import tensorflow as tf
from PyQt5 import QtGui
from PIL.ImageQt import ImageQt
from PIL import ImageOps, Image

class SlImage():
    def __init__(self, filename, bb_height, bb_width):
        self.wsiObj = ops.OpenSlide(filename)
        self.bb_height = bb_height
        self.bb_width = bb_width
        self.level = self.wsiObj.level_count - 1
        self.leveldim = self.wsiObj.level_dimensions
        self.imheight = None
        self.imwidth = None
        self.starth = None
        self.startw = None
        self.curim = None

    def read_first(self):
        self.curim = self.wsiObj.read_region((0,0), self.level, self.wsiObj.level_dimensions[self.level])
        print(self.curim.size)
        self.imheight = self.curim.size[0]
        self.imwidth = self.curim.size[1]
        if self.wsiObj.level_dimensions[self.level][0] < self.bb_height or self.wsiObj.level_dimensions[self.level][1] < self.bb_width:
            pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
            self.starth = int((pim.size[0]-self.curim.size[0])/2)
            self.startw = int((pim.size[1]-self.curim.size[1])/2)
            pim.paste(self.curim, (self.starth, self.startw))
            return ImageQt(pim)
        # return ImageQt(im)

    def get_image(self, factor=2):
        if self.imheight*factor < self.bb_height and self.imwidth*factor < self.bb_width:
            pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
            self.curim = self.curim.resize(factor*self.curim.size, Image.ANTIALIAS)
            self.imheight *= factor
            self.imwidth *= factor
            pim.paste(self.curim, (int((pim.size[0] - self.curim.size[0]) / 2), int((pim.size[1] - self.curim.size[1]) / 2)))
            return ImageQt(pim)
        if self.imheight*factor >= self.bb_height:
            if self.imwidth*factor >= self.bb_width:
                print("Inside Popular condition")
                left = int((self.bb_width) / 4)
                top = int((self.bb_height) / 4)
                right = int(3*(self.bb_width) / 4)
                bottom = int(3*(self.bb_height) / 4)
                self.curim = self.curim.crop((left, top, right, bottom))
                ## Add code to check if another level is needed
                self.curim = self.curim.resize(factor * self.curim.size, Image.ANTIALIAS)
                self.imheight = self.bb_height
                self.imwidth = self.bb_width
                return ImageQt(self.curim)
            else:
                print("I am in height condition")
                pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
                left = 0
                top = int((self.bb_height) / 4)
                right = self.bb_width
                bottom = int(3 * (self.bb_height) / 4)
                self.curim = self.curim.crop((left, top, right, bottom))
                ## Add code to check if another level is needed
                self.curim = self.curim.resize(factor * self.curim.size, Image.ANTIALIAS)
                self.imheight = self.bb_height
                self.imwidth *= factor
                pim.paste(self.curim,(0, int((pim.size[1] - self.curim.size[1]) / 2)))
                return ImageQt(pim)
        if self.imwidth*factor >= self.bb_width:
            print("I am in width condition")
            pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
            left = int(self.bb_width/4)
            top = 0
            right = int((3*self.bb_width)/4)
            bottom = self.bb_height
            self.curim.show()
            self.curim = self.curim.crop((left, top, right, bottom))
            print("Crop completed")
            self.curim.show()
            ## Add code to check if another level is needed
            self.curim = self.curim.resize((2*self.curim.size[1], 2*self.curim.size[0]), Image.ANTIALIAS)
            print("Resizing done")
            self.imheight *= factor
            self.imwidth = self.bb_width
            pim.paste(self.curim, (int((pim.size[0] - self.curim.size[0]) / 2), 0))
            pim.show()
            return ImageQt(pim)

# def main2():
#     # image files
#     overlap = 44
#     wsiPath = cm.wsiPath + cm.wsiName + cm.wsiExt
#     maskImg = cv.imread(cm.bgMaskPath + cm.wsiName + cm.bgMaskExt)
#
#     if (os.path.isdir(cm.wsiName)):
#         print('Directory already exist, the existing data might be overwrite')
#         input("Press Enter to continue...")
#     else:
#         os.mkdir(cm.wsiName, 777);
#
#     print(wsiPath)
#     print(cm.bgMaskPath + cm.wsiName + cm.bgMaskExt)
#
#     # load WSI
#     wsiObj = ops.OpenSlide(wsiPath)
#     width = wsiObj.dimensions[0]
#     height = wsiObj.dimensions[1]
#     numberOfLevels = wsiObj.level_count
#     downsampleFactors = wsiObj.level_downsamples
#
#     width = wsiObj.level_dimensions[cm.resLev][0]
#     height = wsiObj.level_dimensions[cm.resLev][1]
#     probMask = np.zeros([height, width], dtype=np.int8)
#
#     rSt, cSt = 0, 0
#     rEn = height / cm.tilesRow
#     cEn = width / cm.tilesCol
#
#     mrSt, mcSt = 0, 0
#     maskR, maskC = maskImg.shape[:2]
#     mrEn = math.floor(maskR / cm.tilesRow)
#     mcEn = math.floor(maskC / cm.tilesCol)
#
#     for i in range(cm.tilesRow):
#         for j in range(cm.tilesCol):
#             rS = int(round(rSt * downsampleFactors[cm.resLev]))
#             cS = int(round(cSt * downsampleFactors[cm.resLev]))
#
#             if j != cm.tilesCol - 1:
#                 cEnt = cEn + overlap
#                 mcEnt = mcEn
#             else:
#                 cEnt = cEn
#                 mcEnt = mcEn
#
#             if i != cm.tilesRow - 1:
#                 rEnt = rEn + overlap
#                 mrEnt = mcEn
#             else:
#                 rEnt = rEn
#                 mrEnt = mrEn
#
#             # print mcSt, mcEn, mrSt, mrEn
#             maskRoi = maskImg[mrSt:mrSt + mrEnt, mcSt:mcSt + mcEnt]
#             mMean, mStd = cv.meanStdDev(maskRoi)
#
#             if (mMean[0] + mMean[1] + mMean[2] != 0):
#                 print(rSt, cSt, rEnt, cEnt)
#
#                 wsiRoi = wsiObj.read_region((cS, rS), cm.resLev, (int(cEnt), int(rEnt)))
#                 r, g, b, a = cv.split(np.array(wsiRoi))
#                 wsiRoi = cv.merge([b, g, r])
#                 cv.imwrite(cm.wsiName + '/' + cm.wsiName + '_' + str(i * cm.tilesRow + j) + '.jpg', wsiRoi)
#                 # wsiRoi.save(cm.wsiName + '/'+ cm.wsiName+ '_' +str(i*cm.tilesRow + j) + '.jpg', "JPEG")
#                 # wsiRoi =  cv.imread(cm.wsiName + '/' + cm.wsiName+ '_' +str(i*cm.tilesRow + j) + '.jpg', 3)
#                 wsiRoi_h, wsiRoi_w = wsiRoi.shape[:2]
#
#                 r_maskRoi = cv.resize(maskRoi, (wsiRoi_w, wsiRoi_h))
#                 result = CM_test_functions.test(cm.modelPath, wsiRoi, r_maskRoi)
#
#                 outMask = result[:, :, 0]
#                 outMask = outMask * 100
#                 outMask = np.int8(outMask)
#
#                 if j != cm.tilesCol - 1:
#                     outMask = outMask[:, 0:wsiRoi_w - overlap]
#
#                 if i != cm.tilesRow - 1:
#                     outMask = outMask[0:wsiRoi_h - overlap, :]
#
#                 probMask[rSt:rSt + rEn, cSt:cSt + cEn] += outMask
#                 cv.imwrite(cm.wsiName + '/' + cm.wsiName + '_' + str(i * cm.tilesRow + j) + '_m.bmp', outMask)
#
#             cSt = cEn * (j + 1)
#             mcSt = mcEn * (j + 1)
#         rSt = rEn * (i + 1)
#         mrSt = mrEn * (i + 1)
#         cSt = 0
#         mcSt = 0
#     cv.imwrite(cm.wsiName + '/' + cm.wsiName + '_mask.jpg', probMask)