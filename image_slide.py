import numpy as np
import openslide as ops
from PIL import Image
import matlab.engine as me
import matlab
import os

class ImageClass():
    def __init__(self, filename):
        print("Initiating class with ", filename)
        if ".tif" in filename:
            self.wsiObj = ops.OpenSlide(filename)
            self.level_count = self.wsiObj.level_count
            self.level_dimensions = self.wsiObj.level_dimensions
            self.type = "tiff"
        if ".jp2" in filename:
            self.filename = filename
            self.eng = me.start_matlab()
            l, w, h = self.eng.get_info(self.filename, nargout=3)
            self.level_count = int(l)
            self.level_dimensions = []
            for i in range(self.level_count):
                self.level_dimensions.append((int(w), int(h)))
                w, h = int(w/2), int(h/2)
            self.type = "jp2"
        if os.path.isdir(filename):
            self.wsiObj = []
            self.tlist = os.listdir(filename)
            self.level_count = len(self.tlist)
            self.level_dimensions = []
            for i in range(self.level_count):
                self.wsiObj.append(ops.ImageSlide(filename + "/" + self.tlist[i]))
                self.wsiObj[i].read_region((0, 0), 0, (10, 10))
                self.level_dimensions.append(self.wsiObj[i].level_dimensions[0])
            self.type = "png_folder"
        print(self.level_count, self.level_dimensions, type(self.level_dimensions))

    def read_region(self, coor_low, level, dim):
        # print("Reading Region: ", coor_low, level, dim)
        if self.type=="tiff":
            return self.wsiObj.read_region(coor_low, level, dim)
        elif self.type=="jp2":
            coor_cur_w = int(coor_low[0] / pow(2, level))
            coor_cur_h = int(coor_low[1] / pow(2, level))
            hleft = max(1, coor_cur_h + 1)
            hdown = min(self.level_dimensions[level][1], hleft + dim[1])
            wleft = max(1, coor_cur_w + 1)
            wright = min(self.level_dimensions[level][0], wleft + dim[0])
            if dim[0] <= self.level_dimensions[level][0] and dim[1] <= self.level_dimensions[level][1]:
                mim = self.eng.read_region(self.filename, level, matlab.int32([hleft, hdown, wleft, wright]))
                im = np.array(mim._data).reshape(mim.size, order='F')
                return Image.fromarray(im).convert("RGBA")
            else:
                pim = Image.new("RGBA", dim, (255, 255, 255, 0))
                mim = self.eng.read_region(self.filename, level, matlab.int32([hleft, hdown, wleft, wright]))
                im = np.array(mim._data).reshape(mim.size, order='F')
                im = Image.fromarray(im).convert("RGBA")
                startw = int((pim.size[0] - im.size[0]) / 2)
                starth = int((pim.size[1] - im.size[1]) / 2)
                pim.paste(im, (startw, starth))
                return pim
        elif self.type=="png_folder":
            return self.wsiObj[level].read_region(coor_low, 0, dim)
