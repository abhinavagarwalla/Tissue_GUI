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

def main():
    print(1)
    wsiPath = 'data/Tumor_001.tif'
    wsiObj = ops.OpenSlide(wsiPath)
    width = wsiObj.dimensions[0]
    height = wsiObj.dimensions[1]
    numberOfLevels = wsiObj.level_count
    downsampleFactors = wsiObj.level_downsamples
    print(width, height, numberOfLevels, downsampleFactors)
    print(wsiObj.level_dimensions)
    im = wsiObj.read_region((1000,1000), 6, (2000,2000))
    im.show()
    ## print(im)
    # cv.imshow("Image", im)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

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

if __name__ == "__main__":
    main()