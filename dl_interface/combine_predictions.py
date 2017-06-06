import glob

import cv2
import numpy as np
from PIL import Image

from dl_interface.model_config import *
from interface.image_slide import ImageClass


def combine():
    ilist = glob.glob(Config.RESULT_PATH+"\\*)_tumor.png")

    # (48896, 110336)
    wsi = ImageClass(Config.WSI_PATH)
    wsiDim = (wsi.level_dimensions[Config.LEVEL_FETCH][1], wsi.level_dimensions[Config.LEVEL_FETCH][0]) #(110336, 48896)
    arr = np.zeros(wsiDim)
    for i in ilist:
        name = i.split('\\')[-1].split(',')
        h, w = int(name[0][1:])+Config.DIFF_SIZE, int(name[1].split(')')[0])+Config.DIFF_SIZE
        im = np.array(Image.open(i))
        if h+Config.OUTPUT_SIZE>wsiDim[1]:
            if w+Config.OUTPUT_SIZE>wsiDim[0]:
                im = im[:(wsiDim[0]-w), :(wsiDim[1]-h)]
            else:
                im = im[:, :(wsiDim[1]-h)]
        if w+Config.OUTPUT_SIZE>wsiDim[0]:
            im = im[:(wsiDim[0]-w), :]
        arr[w:w+Config.OUTPUT_SIZE, h:h+Config.OUTPUT_SIZE] = im

    print("Writing Image")
    cv2.imwrite(Config.RESULT_PATH + "\\predictions_png\\level_"+str(Config.LEVEL_FETCH)+".png", arr)
    del arr
    pim = Image.open(Config.RESULT_PATH + "\\predictions_png\\level_"+str(Config.LEVEL_FETCH)+".png")
    for i in range(Config.LEVEL_FETCH+1, wsi.level_count):
        pim = pim.resize((int(pim.size[0]/2), int(pim.size[1]/2)))
        pim.save(Config.RESULT_PATH + "\\predictions_png\\level_"+str(i)+".png")
    print("Image Write Completed")