import os
import cv2 as cv
import numpy as np
import openslide as ops

import sys
import math
# import tensorflow as tf
from PIL.ImageQt import ImageQt
from PIL import ImageOps, Image, ImageDraw, ImageChops

class ImageSlide():
    def __init__(self, filename):
        if ".tiff" in filename:
            self.wsiObj = ops.OpenSlide(filename)
            self.level_count = self.wsiObj.level_count
            self.level_dimensions = self.wsiObj.level_dimensions
            self.type = "tiff"

    def read_region(self, coor_low_w, coor_low_h, level, width, height):
        if self.type=="tiff":
            return self.wsiObj.read_region((coor_low_w, coor_low_h), level, (width, height))