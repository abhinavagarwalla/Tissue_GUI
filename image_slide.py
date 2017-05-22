import numpy as np
import openslide as ops
from PIL import Image
import matlab.engine as me
import matlab

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
        print(self.level_count, self.level_dimensions, type(self.level_dimensions))

    def read_region(self, coor_low, level, dim):
        coor_cur_w = int(coor_low[0]/pow(2, level))
        coor_cur_h = int(coor_low[1]/pow(2, level))
        print("Reading Region: ", coor_low, level, dim)
        if self.type=="tiff":
            return self.wsiObj.read_region(coor_low, level, dim)
        elif self.type=="jp2":
            hleft = coor_cur_h + 1
            hleft = max(1, hleft)
            hdown = hleft + dim[1]
            hdown = min(self.level_dimensions[level][1], hdown)
            wleft = coor_cur_w + 1
            wleft = max(1, wleft)
            wright = wleft + dim[0]
            wright = min(self.level_dimensions[level][0], wright)
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
