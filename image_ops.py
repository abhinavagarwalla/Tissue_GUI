import os
import cv2 as cv
import numpy as np
import openslide as ops

import sys
import math
# import tensorflow as tf
from PIL.ImageQt import ImageQt
from PIL import ImageOps, Image
from image_overlay import SegMaskByPixel

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
        print(self.leveldim)

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
            print("Inside the right box")
            self.curim = self.curim.resize((int(self.curim.size[0]/2), int(self.curim.size[1]/2)), Image.ANTIALIAS)
            self.imheight = int(self.imheight/factor)
            self.imwidth = int(self.imwidth/factor)
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

    def pan(self, direction=None, step=0.05):
        if_updated = False
        if direction == 'left':
            if int(self.coor_cur_w - step*self.bb_width) < self.leveldim[self.level][0] and self.coor_cur_w>step*self.bb_width:
                self.coor_cur_w = int(self.coor_cur_w - step*self.bb_width)
                self.coor_low_w = pow(2, self.level) * self.coor_cur_w
                if_updated = True
        if direction == 'right':
            if int(self.coor_cur_w + step*self.bb_width + self.bb_width) < self.leveldim[self.level][0]:
                self.coor_cur_w = int(self.coor_cur_w + step*self.bb_width)
                self.coor_low_w = pow(2, self.level) * self.coor_cur_w
                if_updated = True
        if direction == 'up':
            if int(self.coor_cur_h - step*self.bb_height) < self.leveldim[self.level][1] and self.coor_cur_h>step*self.bb_height:
                self.coor_cur_h = int(self.coor_cur_h - step*self.bb_height)
                self.coor_low_h = pow(2, self.level) * self.coor_cur_h
                if_updated = True
        if direction == 'down':
            if int(self.coor_cur_h + step*self.bb_height + self.bb_height) < self.leveldim[self.level][1]:
                self.coor_cur_h = int(self.coor_cur_h + step*self.bb_height)
                self.coor_low_h = pow(2, self.level) * self.coor_cur_h
                if_updated = True
        if direction:
            self.curim = self.wsiObj.read_region((self.coor_low_w, self.coor_low_h), self.level,
                                                 (self.bb_width, self.bb_height))
        return ImageQt(self.curim), if_updated

    def read_first_overlay(self, filename, method=None, method_update="init"):
        print(method)
        if method==0:
            print("Inside Segmentation")
            self.overlayObj = SegMaskByPixel(filename, self.wsiObj, self.bb_height, self.bb_width)
            self.overlayim = self.overlayObj.get_overlay(self.level, self.coor_cur_w, self.coor_cur_h, self.imwidth,
                                                         self.imheight, method_update)
            self.overlay_on_orig_image()
            return ImageQt(self.overlayim)

    def update_overlay(self, method_update="init", step=None):
        self.overlayim = self.overlayObj.get_overlay(self.level, self.coor_cur_w, self.coor_cur_h, self.imwidth,
                                                     self.imheight, method_update, step)
        self.overlay_on_orig_image()
        return ImageQt(self.overlayim)

    def overlay_on_orig_image(self):
        self.overlayim = Image.blend(self.curim, self.overlayim, 0.5)