import openslide as ops
import numpy as np
from multiprocessing import Pool
from itertools import permutations
from functools import partial
# from pathos.multiprocessing import ProcessingPool as Pool
from PIL import Image

class SegMaskByPixel():
    def __init__(self, filename, wsiObj, bb_height, bb_width):
        if 'tif' in filename:
            try:
                self.ovObj = ops.OpenSlide(filename)
                self.type = "OpenSlide"
                print("OpenSlide Object Read")
                self.nlevel = self.ovObj.level_count
                self.leveldim = self.ovObj.level_dimensions
            except:
                try:
                    self.ovObj = ops.ImageSlide(filename)
                    self.type = "Image"
                except:
                    print("Undefined Behaviour or Unsupported File Format")
        else:
            # self.ovObj = Image.open(filename) ##Normal Image Reader
            self.ovObj = ops.ImageSlide(filename)
            self.type = "Image"
            print("Image has been read")
            self.nlevel = self.ovObj.level_count
            self.leveldim = self.ovObj.level_dimensions
            print(self.nlevel, self.leveldim)
        self.bb_height = bb_height
        self.bb_width = bb_width
        self.clevel, self.olevel = None, None
        self.hratio, self.wratio = None, None
        self.wsidim = [wsiObj.level_dimensions[i] for i in range(len(wsiObj.level_dimensions))]
        self.levelfetch = 2
        self.overlayim = None

    def get_overlay_simple(self, level, coorw, coorh, width, height):
        coor_low_w = pow(2, level) * coorw
        coor_low_h = pow(2, level) * coorh

        im = self.ovObj.read_region((coor_low_w, coor_low_h), level, (width, height))
        return im

    def get_overlay_parallel(self, level, coorw, coorh, width, height):
        print("Inside Parallel Implementation")
        self.coor_low_w = pow(2, level) * coorw
        self.coor_low_h = pow(2, level) * coorh
        self.low_width = pow(2, level - self.levelfetch) * width
        self.low_height = pow(2, level - self.levelfetch) * height
        # print(coor_low_h, coor_low_w)
        # print(low_height, low_width)
        # print(level, self.levelfetch)
        # print(self.leveldim[self.levelfetch])
        # print(self.leveldim
        pstep = 1024
        p = Pool()
        # ws = range(0, low_width, self.pstep)
        ps = list(permutations(range(0, self.low_width, pstep), 2))
        print(type(ps))
        print(ps)
        print(self.low_height, self.low_width)
        # func = partial(self.get_simple_region, ovObj, coor_low_w, coor_low_h, levelfetch, low_width,
        #                low_height)
        imList = p.map(self, ps)
        imList[0].show()
        # imList = process_whole(self.ovObj, self.coor_low_w, self.coor_low_h, self.levelfetch, self.low_width, self.low_height, width, height)
        # imList = p.map(func, ps)
        # print(len(imList))
        return imList

    def get_overlay(self, level, coorw, coorh, width, height, method=None, step=None):
        if self.type=="OpenSlide":
            return self.get_overlay_openslide(level, coorw, coorh, width, height, method, step)
        elif self.type=="Image":
            print("Reading as Image format")
            return self.get_overlay_image(level, coorw, coorh, width, height, method, step)
        else:
            print("Unsupported File Format")
            return 1

    def get_overlay_image(self, level, coorw, coorh, width, height, method, step):
        # print("Inside Overlay")
        # if self.nlevel==1:
        # # if True:
        #     print("Number of levels encountered is 1")
        #     level_diff = int(np.log2(int(self.wsidim[0][0]/self.leveldim[self.levelfetch][0]))) #replace levelfetch by 0
        #     print("Level difference is=", level_diff, "with levels=", level)
        #     coor_low_w = pow(2, level - level_diff) * coorw
        #     coor_low_h = pow(2, level - level_diff) * coorh
        #     low_width = pow(2, level - level_diff - self.levelfetch) * width
        #     low_height = pow(2, level - level_diff - self.levelfetch) * height
        #     im = self.ovObj.read_region((coor_low_w, coor_low_h), self.levelfetch, (low_width, low_height))
        #     # print(im.size)
        #     im.show()
        #     im = im.resize((width, height))
        #     return im

        return self.overlayim

    def get_overlay_openslide(self, level, coorw, coorh, width, height, method=None, step=None):
        ## Currenly leaving out: Case in which a level0, level1 segmask are left out
        ## For such case, make use of level fetch
        ## Fix cases for which level >= self.nlevel
        if level < self.nlevel:
            print(self.wsidim[level])
            print(self.leveldim[level])
            if self.wsidim[level]==self.leveldim[level]:
                print("Inside Simple Overlay")
                return self.get_overlay_simple(level, coorw, coorh, width, height)
            # Write here as else condition for handling missing alignment cases
        if level >= self.nlevel:
            self.levelfetch = self.nlevel-1

        self.coor_low_w = pow(2, level) * coorw
        self.coor_low_h = pow(2, level) * coorh
        self.cur_width = pow(2, level - self.levelfetch) * width
        self.cur_height = pow(2, level - self.levelfetch) * height
        self.low_width = pow(2, level) * width
        self.low_height = pow(2, level) * height
        print("Getting Overlay ", method)
        if method=="init":
            print("Inside init method")
            self.overlayim = self.ovObj.read_region((self.coor_low_w, self.coor_low_h), self.levelfetch,
                                                    (self.cur_width, self.cur_height))
            self.overlayim = self.overlayim.resize((width, height))
            return self.overlayim
        elif method=="zoom_in":
            left = int(self.bb_width/4)
            top = int(self.bb_height/4)
            right = int(3*self.bb_width / 4)
            bottom = int(3*self.bb_height / 4)
            self.overlayim = self.overlayim.crop((left, top, right, bottom))
            # tim = self.ovObj.read_region((self.coor_low_w, self.coor_low_h), self.levelfetch,
            #                                         (self.cur_width, self.cur_height))
            self.overlayim = self.overlayim.resize((width, height))
            return self.overlayim
        elif method=="zoom_out":
            pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
            self.overlayim = self.overlayim.resize((int(self.bb_width/2), int(self.bb_height/2)))
            pim.paste(self.overlayim, (int(self.bb_width/4), int(self.bb_height/4)))
            # pim.show()
            #Now read 4 surrounding regions
            tim_top = self.ovObj.read_region((self.coor_low_w, self.coor_low_h), self.levelfetch,
                                        (self.cur_width, int(self.cur_height/4)))
            tim_top = tim_top.resize((self.bb_width, int(self.bb_height/4)))
            pim.paste(tim_top, (0, 0))

            tim_bottom = self.ovObj.read_region((self.coor_low_w, self.coor_low_h + int(3*self.low_height/4)), self.levelfetch,
                                             (self.cur_width, int(self.cur_height / 4)))
            tim_bottom = tim_bottom.resize((self.bb_width, int(self.bb_height / 4)))
            pim.paste(tim_bottom, (0, int(3*self.bb_height/4)))

            tim_left = self.ovObj.read_region((self.coor_low_w, self.coor_low_h + int(self.low_height/4)), self.levelfetch,
                                             (int(self.cur_width/4), int(self.cur_height / 2)))
            tim_left = tim_left.resize((int(self.bb_width/4), int(self.bb_height / 2)))
            pim.paste(tim_left, (0, int(self.bb_height/4)))

            tim_right = self.ovObj.read_region((self.coor_low_w + int(3*self.low_width/4), self.coor_low_h + int(self.low_height/4)),
                                              self.levelfetch, (int(self.cur_width / 4), int(self.cur_height / 2)))
            tim_right = tim_right.resize((int(self.bb_width / 4), int(self.bb_height / 2)))
            pim.paste(tim_right, (int(3*self.bb_width/4), int(self.bb_height / 4)))

            # self.ovObj.read_region((self.coor_low_w, self.coor_low_h), self.levelfetch,
            #                             (self.cur_width, self.cur_height)).show()
            self.overlayim = pim
            return self.overlayim
        if step:
            if method=="left":
                print("Inside left method ", step)
                pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
                self.overlayim = self.overlayim.crop((0, 0, int((1-step)*self.bb_width), self.bb_height))
                pim.paste(self.overlayim, (int(step*self.bb_width), 0))
                tim = self.ovObj.read_region((self.coor_low_w, self.coor_low_h), self.levelfetch,
                                             (int(self.cur_width*step), self.cur_height))
                tim = tim.resize((int(step*self.bb_width), self.bb_height))
                pim.paste(tim, (0, 0))
                self.overlayim = pim
                return self.overlayim
            elif method=="right":
                print("Inside right method ", step)
                pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
                self.overlayim = self.overlayim.crop((int(step*self.bb_width), 0, self.bb_width, self.bb_height))
                pim.paste(self.overlayim, (0, 0))
                tim = self.ovObj.read_region((self.coor_low_w + int((1-step)*self.low_width), self.coor_low_h),
                                             self.levelfetch, (int(self.cur_width * step), self.cur_height))
                tim = tim.resize((int(step*self.bb_width), self.bb_height))
                pim.paste(tim, (int((1-step)*self.bb_width), 0))
                self.overlayim = pim
                return self.overlayim
            if method=="up":
                print("Inside left method ", step)
                pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
                self.overlayim = self.overlayim.crop((0, 0, self.bb_width, int((1-step)*self.bb_height)))
                pim.paste(self.overlayim, (0, int(step*self.bb_height)))
                tim = self.ovObj.read_region((self.coor_low_w, self.coor_low_h), self.levelfetch,
                                             (self.cur_width, int(step * self.cur_height)))
                tim = tim.resize((self.bb_width, int(step*self.bb_height)))
                pim.paste(tim, (0, 0))
                self.overlayim = pim
                return self.overlayim
            if method=="down":
                print("Inside left method ", step)
                pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
                self.overlayim = self.overlayim.crop((0, int(step*self.bb_height), self.bb_width, self.bb_height))
                pim.paste(self.overlayim, (0, 0))
                tim = self.ovObj.read_region((self.coor_low_w, self.coor_low_h + int((1-step)*self.low_height)), self.levelfetch,
                                             (self.cur_width, int(step * self.cur_height)))
                tim = tim.resize((self.bb_width, int(step*self.bb_height)))
                pim.paste(tim, (0, int((1-step)*self.bb_height)))
                self.overlayim = pim
                return self.overlayim