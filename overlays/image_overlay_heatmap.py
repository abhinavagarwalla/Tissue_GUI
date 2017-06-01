import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from image_slide import ImageClass


class HeatMap():
    def __init__(self, filename, wsiObj, bb_height, bb_width):
        self.wsidim = [wsiObj.level_dimensions[i] for i in range(len(wsiObj.level_dimensions))]
        self.ovObj = ImageClass(filename)
        self.level_fetch = 0
        for i in range(len(self.wsidim)):
            if self.wsidim[i] == self.ovObj.level_dimensions[0]:
                self.level_fetch = i
                continue

        self.type = "Image"
        self.overlayim = None
        self.cmap = plt.get_cmap("jet")

    def get_overlay(self, level, coorw, coorh, width, height, method=None, step=None, class_states=None):
        print("Getting Simple Overlay", self.level_fetch, level, coorw, coorh, width, height)
        if level >= self.level_fetch:
            self.overlayim = self.ovObj.read_region((coorw, coorh), level - self.level_fetch, (width, height))
        else:
            level_diff = level - self.level_fetch
            coor_low_w = int(pow(2, level_diff) * coorw)
            coor_low_h = int(pow(2, level_diff) * coorh)
            width_low = int(pow(2, level_diff) * width)
            height_low = int(pow(2, level_diff) * height)
            self.overlayim = self.ovObj.read_region((coor_low_w, coor_low_h), 0, (width_low, height_low))
            self.overlayim = self.overlayim.resize((width, height))

        self.overlayim = np.array(self.overlayim.convert("L")) / 99
        self.overlayim = np.uint8(self.cmap(self.overlayim) * 255)
        self.overlayim = Image.fromarray(self.overlayim)
        bands = list(self.overlayim.split())
        bands[3] = bands[3].point(lambda x: x * 0.2)
        self.overlayim = Image.merge(self.overlayim.mode, bands)
        return self.overlayim
