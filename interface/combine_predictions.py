import glob

import cv2
import numpy as np
import openslide as ops
from PIL import Image
from interface.model_config import *


def combine():
    RESULT_PATH = 'C:\\Users\\abhinav\\Desktop\\Tissue_GUI\\results\\'
    ilist = glob.glob(RESULT_PATH+"*)_tumor.png")

    # (48896, 110336)
    wsi = ops.OpenSlide(WSI_PATH)
    wsiDim = (wsi.level_dimensions[LEVEL_FETCH][1], wsi.level_dimensions[LEVEL_FETCH][0]) #(110336, 48896)
    arr = np.zeros(wsiDim)
    for i in ilist:
        name = i.split('\\')[-1].split(',')
        h, w = int(name[0][1:])+DIFF_SIZE, int(name[1].split(')')[0])+DIFF_SIZE
        im = np.array(Image.open(i))
        if h+OUTPUT_SIZE>wsiDim[1]:
            if w+OUTPUT_SIZE>wsiDim[0]:
                im = im[:(wsiDim[0]-w), :(wsiDim[1]-h)]
            else:
                im = im[:, :(wsiDim[1]-h)]
        if w+OUTPUT_SIZE>wsiDim[0]:
            im = im[:(wsiDim[0]-w), :]
        arr[w:w+OUTPUT_SIZE, h:h+OUTPUT_SIZE] = im

    print("Writing Image")
    cv2.imwrite("C:\\Users\\abhinav\\Desktop\\Tissue_GUI\\predictions_png\\level_"+str(LEVEL_FETCH)+".png", arr)
    del arr
    pim = Image.open("C:\\Users\\abhinav\\Desktop\\Tissue_GUI\\predictions_png\\level_"+str(LEVEL_FETCH)+".png")
    for i in range(LEVEL_FETCH+1, wsi.level_count):
        pim = pim.resize((int(pim.size[0]/2), int(pim.size[1]/2)))
        pim.save("C:\\Users\\abhinav\\Desktop\\Tissue_GUI\\predictions_png\\level_"+str(i)+".png")
    print("Image Write Completed")