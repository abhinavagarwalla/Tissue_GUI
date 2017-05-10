import openslide as ops

class SegMaskByPixel():
    def __init__(self, filename, bb_height, bb_width):
        self.wsiObj = ops.OpenSlide(filename)
        self.bb_height = bb_height
        self.bb_width = bb_width
        self.nlevel = self.wsiObj.level_count
        self.leveldim = self.wsiObj.level_dimensions
        print(self.nlevel)
        print(self.leveldim)

    def get_overlay(self, level, coorw, coorh, width, height):
        coor_low_w = pow(2, level) * coorw
        coor_low_h = pow(2, level) * coorh

        im = self.wsiObj.read_region((coor_low_w, coor_low_h), level, (width, height))
        # self.wsiObj.read_region((0,0), 0, self.leveldim[0]).show()
        return im