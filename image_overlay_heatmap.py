import openslide as ops
import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat
from shapely.geometry import Polygon, MultiPolygon
import cv2 as cv
import h5py
import matplotlib.pyplot as plt
import os

class HeatMap():
    def __init__(self, filename, wsiObj, bb_height, bb_width):
        self.wsidim = [wsiObj.level_dimensions[i] for i in range(len(wsiObj.level_dimensions))]
        self.ovObj = []
        self.level_fetch = 0
        tlist = os.listdir(filename)
        for i in range(len(tlist)):
            print(tlist[i])
            self.ovObj.append(ops.ImageSlide(filename + "/" + tlist[i]))
            self.ovObj[i].read_region((0,0), 0, (10,10))

        for i in range(len(self.wsidim)):
            if self.wsidim[i] == self.ovObj[0].level_dimensions[0]:
                self.level_fetch = i
                continue

        self.type = "Image"
        self.leveldim = self.wsidim
        self.bb_height = bb_height
        self.bb_width = bb_width
        self.clevel, self.olevel = None, None
        self.hratio, self.wratio = None, None
        self.overlayim = None
        self.cmap = plt.get_cmap("YlOrRd")

    def get_overlay_simple(self, level, coorw, coorh, width, height):
        print("Getting Simple Overlay", self.level_fetch, level, coorw, coorh, width, height)
        if level >= self.level_fetch:
            self.overlayim = self.ovObj[level-self.level_fetch].read_region((coorw, coorh), 0, (width, height))
        else:
            level_diff = level - self.level_fetch
            coor_low_w = int(pow(2, level_diff) * coorw)
            coor_low_h = int(pow(2, level_diff) * coorh)
            width_low = int(pow(2, level_diff) * width)
            height_low = int(pow(2, level_diff) * height)
            self.overlayim = self.ovObj[0].read_region((coor_low_w, coor_low_h), 0, (width_low, height_low))
            self.overlayim = self.overlayim.resize((width, height))

        self.overlayim = np.array(self.overlayim.convert("L"))/99
        pim = np.uint8(self.cmap(self.overlayim)*255)
        self.overlayim = Image.fromarray(pim)
        return self.overlayim

    def get_overlay(self, level, coorw, coorh, width, height, method=None, step=None):
        print("Getting overlay")
        return self.get_overlay_simple(level, coorw, coorh, width, height)