import openslide as ops
import numpy as np
from multiprocessing import Pool
from itertools import permutations
from functools import partial
# from pathos.multiprocessing import ProcessingPool as Pool

class SegMaskByPixel():
    def __init__(self, filename, wsiObj, bb_height, bb_width):
        print("Reading as openslide")
        self.ovObj = ops.OpenSlide(filename)
        print("Read as openslide")
        self.bb_height = bb_height
        self.bb_width = bb_width
        self.nlevel = self.ovObj.level_count
        self.leveldim = self.ovObj.level_dimensions
        self.clevel, self.olevel = None, None
        self.hratio, self.wratio = None, None
        self.wsidim = [wsiObj.level_dimensions[i] for i in range(len(wsiObj.level_dimensions))]
        self.levelfetch = 2

    def get_overlay_simple(self, level, coorw, coorh, width, height):
        coor_low_w = pow(2, level) * coorw
        coor_low_h = pow(2, level) * coorh

        im = self.ovObj.read_region((coor_low_w, coor_low_h), level, (width, height))
        return im

    def __call__(self, c):
        print("Reading Region: ")
        return self.ovObj.read_region((self.coor_low_w, self.coor_low_h), self.levelfetch,
                                 (self.low_width + c[0], self.low_height + c[1]))

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

    def get_overlay(self, level, coorw, coorh, width, height):
        if level < self.nlevel:
            print(self.wsidim[level])
            print(self.leveldim[level])
            if self.wsidim[level]==self.leveldim[level]:
                print("Inside Simple Overlay")
                return self.get_overlay_simple(level, coorw, coorh, width, height)

        print("Inside Overlay")
        if self.nlevel==1:
        # if True:
            print("Number of levels encountered is 1")
            level_diff = int(np.log2(int(self.wsidim[0][0]/self.leveldim[self.levelfetch][0]))) #replace levelfetch by 0
            print("Level difference is=", level_diff, "with levels=", level)
            coor_low_w = pow(2, level - level_diff) * coorw
            coor_low_h = pow(2, level - level_diff) * coorh
            low_width = pow(2, level - level_diff - self.levelfetch) * width
            low_height = pow(2, level - level_diff - self.levelfetch) * height
            im = self.ovObj.read_region((coor_low_w, coor_low_h), self.levelfetch, (low_width, low_height))
            # print(im.size)
            im.show()
            im = im.resize((width, height))
            return im

        # if True:
            # return self.get_overlay_parallel(level, coorw, coorh, width, height)

        coor_low_w = pow(2, level) * coorw
        coor_low_h = pow(2, level) * coorh
        low_width = pow(2, level-self.levelfetch) * width
        low_height = pow(2, level-self.levelfetch) * height
        im = self.ovObj.read_region((coor_low_w, coor_low_h), self.levelfetch, (low_width, low_height))
        im = im.resize((width, height))
        return im

# def get_simple_region(ovObj, coor_low_w, coor_low_h, levelfetch, low_width, low_height, a, b):
#     print("Reading Region: ")
#     return ovObj.read_region((coor_low_w, coor_low_h), levelfetch,
#                                   (low_width + a, low_height + b))
#
# def process_whole(ovObj, coor_low_w, coor_low_h, levelfetch, low_width, low_height, width, height):
#     # imList = get_simple_region(self.ovObj, self.coor_low_w, self.coor_low_h, self.levelfetch, self.low_width, self.low_height, 0, 0)
#     # imList.show()
#     func = partial(get_simple_region, ovObj, coor_low_w, coor_low_h, levelfetch, low_width,
#                    low_height)
#     pstep = 1024
#     p = Pool()
#     # ws = range(0, low_width, self.pstep)
#     ps = list(permutations(range(0, low_width, pstep), 2))
#     print(type(ps))
#     print(ps)
#     print(low_height, low_width)
#     imList = p.starmap(func, ps)
#     print("Returning from starmap")
#     imList = imList[0].resize((width, height))
#     imList.show()
#     return imList