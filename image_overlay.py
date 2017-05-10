import openslide as ops

class SegMaskByPixel():
    def __init__(self, filename, bb_height, bb_width):
        self.wsiObj = ops.OpenSlide(filename)
        self.bb_height = bb_height
        self.bb_width = bb_width
        self.nlevel = self.wsiObj.level_count
        self.leveldim = self.wsiObj.level_dimensions

    def get_overlay(self, level, coorw, coorh, width, height):
        centerh = coorh + height / 2
        centerw = coorw + width / 2
        left = int(centerw - self.bb_width / 4)
        top = int(centerh - self.bb_height / 4)
        coor_low_w = pow(2, level)*left
        coor_low_h = pow(2, level)*top

        im = self.wsiObj.read_region((coor_low_w, coor_low_h), 0, (width, height))
        # self.wsiObj.read_region((0,0), 0, self.leveldim[0]).show()
        return im