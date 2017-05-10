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
        self.imheight, self.imwidth = None, None                ##width, height for the current view
        self.starth, self.startw = None, None                   ##Start h,m for the current view
        self.curim = None                                       ##Stores the current view of the image
        self.orim = None                                        ##Here the lowest level image stored and displayed
        self.zoomlevel = None                                   ##For storing the current zoom level
        self.coor_cur_h, self.coor_cur_w = None, None           ##Actual coordinates in the current view
        self.coor_low_h, self.coor_low_w = None, None           ##Actual coordinates in the lowest view

    def read_first(self):
        self.curim = self.wsiObj.read_region((0,0), self.level, self.wsiObj.level_dimensions[self.level])
        self.orim = self.curim.copy()
        print(self.curim.size)

        self.imheight = self.curim.size[1]
        self.imwidth = self.curim.size[0]
        self.zoomlevel = 1.0
        self.coor_cur_h, self.coor_cur_w = 0, 0
        self.coor_low_h, self.coor_low_w = 0, 0
        if self.wsiObj.level_dimensions[self.level][1] < self.bb_height or self.wsiObj.level_dimensions[self.level][0] < self.bb_width:
            pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
            self.startw = int((pim.size[0]-self.curim.size[0])/2)
            self.starth = int((pim.size[1]-self.curim.size[1])/2)
            pim.paste(self.curim, (self.startw, self.starth))
            # pim.show()
            return ImageQt(self.orim), ImageQt(pim)
        return ImageQt(self.curim)

    def get_image_in(self, factor=2):
        self.zoomlevel *= factor
        if self.imheight*factor < self.bb_height and self.imwidth*factor < self.bb_width:
            pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
            self.curim = self.curim.resize((2*self.curim.size[0], 2*self.curim.size[1]), Image.ANTIALIAS)
            self.imheight *= factor
            self.imwidth *= factor
            self.startw = int((pim.size[0] - self.curim.size[0]) / 2)
            self.starth = int((pim.size[1] - self.curim.size[1]) / 2)
            pim.paste(self.curim, (self.startw, self.starth))
            return ImageQt(pim)
        if self.imheight*factor >= self.bb_height:
            if self.imwidth*factor >= self.bb_width:
                print("Inside Popular condition")
                centerh = self.coor_cur_h + self.imheight / 2
                centerw = self.coor_cur_w + self.imwidth / 2
                left = int(centerw - self.bb_width/4)
                top = int(centerh - self.bb_height/4)

                if self.level:
                    self.level -= 1
                    self.coor_cur_h = 2 * top
                    self.coor_cur_w = 2 * left
                    self.coor_low_h = int(pow(2, self.level) * self.coor_cur_h)
                    self.coor_low_w = int(pow(2, self.level) * self.coor_cur_w)
                    # self.curim.show()
                    self.curim = self.wsiObj.read_region((self.coor_low_w, self.coor_low_h), self.level,
                                                         (self.bb_width, self.bb_height))
                    self.imheight, self.imwidth = self.bb_height, self.bb_width
                    self.starth, self.startw = 0, 0
                    return ImageQt(self.curim)
                else:
                    return ImageQt(self.curim)
            else:
                print("I am in height condition")
                pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
                centerh = self.imheight/2
                left = 0
                top = int(centerh - self.bb_height/ 4)
                if self.level:
                    self.level -= 1
                    self.coor_cur_h = 2 * top
                    self.coor_cur_w = 2 * left
                    self.coor_low_h = int(pow(2, self.level) * self.coor_cur_h)
                    self.coor_low_w = int(pow(2, self.level) * self.coor_cur_w)
                    self.curim = self.wsiObj.read_region((self.coor_low_w, self.coor_low_h), self.level, (self.imwidth*2, self.bb_height))

                    self.imheight = self.bb_height
                    self.imwidth *= 2
                    self.startw = int((pim.size[0] - self.curim.size[0]) / 2)
                    self.starth = 0
                    pim.paste(self.curim,(self.startw, 0))
                    return ImageQt(pim)
                else:
                    return ImageQt(self.curim)
        if self.imwidth*factor >= self.bb_width:
            print("I am in width condition")
            print(self.imwidth, self.bb_width)
            pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
            centerw = self.imwidth/2
            left = centerw - int(self.bb_width/4)
            top = 0
            if self.level:
                self.level -= 1
                self.coor_cur_h = 2 * top
                self.coor_cur_w = 2 * left
                self.coor_low_h = int(pow(2, self.level) * self.coor_cur_h)
                self.coor_low_w = int(pow(2, self.level) * self.coor_cur_w)
                self.curim = self.wsiObj.read_region((self.coor_low_w, self.coor_low_h), self.level,
                                                     (self.bb_width, 2*self.im_height))

                self.imheight *= 2
                self.imwidth = self.bb_width
                self.startw = 0
                self.starth = int((pim.size[1] - self.curim.size[1]) / 2)
                pim.paste(self.curim, (0, self.starth))
                return ImageQt(pim)
            else:
                return ImageQt(self.curim)

    def get_image_out(self, factor=2):
        self.zoomlevel /= factor
        if self.imheight < self.bb_height and self.imwidth < self.bb_width:
            pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
            self.curim = self.curim.resize((self.curim.size[0]/2, self.curim.size[1]/2), Image.ANTIALIAS)
            self.imheight /= factor
            self.imwidth /= factor
            self.startw = int((pim.size[0] - self.curim.size[0]) / 2)
            self.starth = int((pim.size[1] - self.curim.size[1]) / 2)
            pim.paste(self.curim, (self.startw, self.starth))
            return ImageQt(pim)
        if self.imheight >= self.bb_height:
            if self.imwidth >= self.bb_width:
                print("Out: Inside Popular condition")
                centerh = self.coor_cur_h + self.imheight / 2
                centerw = self.coor_cur_w + self.imwidth / 2
                if self.level!=self.wsiObj.level_count-1:
                    self.level += 1
                    self.coor_cur_h = int(centerh/2 - self.bb_height/2)
                    self.coor_cur_w = int(centerw/2 - self.bb_width/2)
                    self.coor_low_h = int(pow(2, self.level) * self.coor_cur_h)
                    self.coor_low_w = int(pow(2, self.level) * self.coor_cur_w)
                    self.curim = self.wsiObj.read_region((self.coor_low_w, self.coor_low_h), self.level,
                                                         (self.bb_width, self.bb_height))
                    print("REgion Processing Complete")
                    self.imheight, self.imwidth = self.bb_height, self.bb_width
                    self.starth, self.startw = 0, 0
                    return ImageQt(self.curim)
                else:
                    return ImageQt(self.curim)
            else:
                print("I am in height condition")
                pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
                centerh = self.coor_cur_h + self.imheight / 2
                centerw = self.coor_cur_w + self.imwidth / 2
                if self.level != self.wsiObj.level_count - 1:
                    self.level += 1
                    self.coor_cur_h = int(centerh / 2 - self.bb_height / 2)
                    self.coor_cur_w = int(centerw / 2 - self.imwidth / 4)
                    self.coor_low_h = pow(2, self.level) * self.coor_cur_h
                    self.coor_low_w = pow(2, self.level) * self.coor_cur_w
                    self.curim = self.wsiObj.read_region((self.coor_low_w, self.coor_low_h), self.level, (self.imwidth/2, self.bb_height))

                    self.imheight = self.bb_height
                    self.imwidth /= 2
                    self.startw = int((pim.size[0] - self.curim.size[0]) / 2)
                    self.starth = 0
                    pim.paste(self.curim,(self.startw, 0))
                    return ImageQt(pim)
                else:
                    return ImageQt(self.curim)
        if self.imwidth >= self.bb_width:
            print("I am in width condition")
            print(self.imwidth, self.bb_width)
            pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
            centerh = self.coor_cur_h + self.imheight / 2
            centerw = self.coor_cur_w + self.imwidth / 2

            if self.level != self.wsiObj.level_count - 1:
                self.level += 1
                self.coor_cur_h = int(centerh / 2 - self.imheight / 4)
                self.coor_cur_w = int(centerw / 2 - self.bb_width / 2)
                self.coor_low_h = pow(2, self.level) * self.coor_cur_h
                self.coor_low_w = pow(2, self.level) * self.coor_cur_w
                self.curim = self.wsiObj.read_region((self.coor_low_w, self.coor_low_h), self.level,
                                                     (self.bb_width, self.imheight/2))

                self.imheight /= 2
                self.imwidth = self.bb_width
                self.startw = 0
                self.starth = int((pim.size[1] - self.curim.size[1]) / 2)
                pim.paste(self.curim, (0, self.starth))
                return ImageQt(pim)
            else:
                return ImageQt(self.curim)

    def get_image_in_temp(self, factor=2):
        self.zoomlevel *= factor
        if self.imheight*factor < self.bb_height and self.imwidth*factor < self.bb_width:
            pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
            self.curim = self.curim.resize((2*self.curim.size[0], 2*self.curim.size[1]), Image.ANTIALIAS)
            self.imheight *= factor
            self.imwidth *= factor
            self.startw = int((pim.size[0] - self.curim.size[0]) / 2)
            self.starth = int((pim.size[1] - self.curim.size[1]) / 2)
            pim.paste(self.curim, (self.startw, self.starth))
            return ImageQt(pim)
        if self.imheight*factor >= self.bb_height:
            if self.imwidth*factor >= self.bb_width:
                print("Inside Popular condition")
                centerh = self.imheight / 2
                centerw = self.imwidth / 2
                left = centerw - int((self.bb_width) / 4)
                top = centerh - int((self.bb_height) / 4)
                right = centerw + int(self.bb_width / 4)
                bottom = centerh + int(self.bb_height / 4)
                self.curim = self.curim.crop((left, top, right, bottom))
                ## Add code to check if another level is needed
                self.curim = self.curim.resize((self.bb_width, self.bb_height), Image.ANTIALIAS)
                self.imheight, self.imwidth = self.bb_height, self.bb_width
                self.starth, self.startw = 0, 0
                return ImageQt(self.curim)
            else:
                print("I am in height condition")
                pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
                centerh = self.imheight/2
                left = 0
                top = centerh - int((self.bb_height) / 4)
                right = self.imwidth
                bottom = centerh + int((self.bb_height) / 4)
                self.curim = self.curim.crop((left, top, right, bottom))
                print("Proceed to cropping")
                ## Add code to check if another level is needed
                self.curim = self.curim.resize((2*self.curim.size[0], self.bb_height), Image.ANTIALIAS)
                self.imheight = self.bb_height
                self.imwidth *= factor
                self.startw = int((pim.size[0] - self.curim.size[0]) / 2)
                self.starth = 0
                pim.paste(self.curim,(self.startw, 0))
                return ImageQt(pim)
        if self.imwidth*factor >= self.bb_width:
            print("I am in width condition")
            print(self.imwidth, self.bb_width)
            pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
            centerw = self.imwidth/2
            left = centerw - int(self.bb_width/4)
            top = 0
            right = centerw + int(self.bb_width/4)
            bottom = self.imheight
            self.curim.show()
            self.curim = self.curim.crop((left, top, right, bottom))
            ## Add code to check if another level is needed
            self.curim = self.curim.resize((self.bb_width, 2*self.curim.size[1]), Image.ANTIALIAS)
            print("Resizing done")
            self.imheight *= factor
            self.imwidth = self.bb_width
            self.startw = 0
            self.starth = int((pim.size[1] - self.curim.size[1]) / 2)
            pim.paste(self.curim, (0, self.starth))
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