import openslide as ops
import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat
from shapely.geometry import Polygon, MultiPolygon

class TumorRegion():
    def __init__(self, filename, wsiObj, bb_height, bb_width):
        self.ovObj = loadmat(filename)["predictions"]
        self.bb_height = bb_height
        self.bb_width = bb_width
        self.wsidim = [wsiObj.level_dimensions[i] for i in range(len(wsiObj.level_dimensions))]
        self.overlayim = None
        self.low_polygons = [Polygon(i[0]) for i in self.ovObj]
        self.npolygons = len(self.low_polygons)

    def get_overlay(self, level, coorw, coorh, width, height, method=None, step=None):
        self.coor_low_w = pow(2, level-1) * coorw
        self.coor_low_h = pow(2, level-1) * coorh
        self.low_width = pow(2, level-1) * width
        self.low_height = pow(2, level-1) * height
        imp = Polygon([(self.coor_low_h, self.coor_low_w), (self.coor_low_h+self.low_height, self.coor_low_w),
                       (self.coor_low_h+self.low_height, self.coor_low_w+self.low_width),
                       (self.coor_low_h, self.coor_low_w + self.low_width)])
        print(imp.bounds)
        print([i.bounds for i in self.low_polygons])
        if_intersect = np.zeros(self.npolygons, dtype=bool)
        region_intersect = []
        for i in range(self.npolygons):
            if imp.intersects(self.low_polygons[i]):
                if_intersect[i] = True
                region_intersect.append(imp.intersection(self.low_polygons[i]))
            else:
                region_intersect.append(None)
        # rPolygons = []
        print(if_intersect)
        pim = Image.new("RGBA", (self.bb_width, self.bb_height), (0, 0, 0, 0))
        for i in range(self.npolygons):
            if if_intersect[i]:
                print(type(region_intersect[i]))
                if isinstance(region_intersect[i], MultiPolygon):
                    print("Handling MultiPolygon Case")
                    for kp in range(len(region_intersect[i])):
                        pcoors = np.array(np.array(region_intersect[i][kp].exterior.coords)/pow(2, level)).astype(int)
                        print("Now drawing on image")
                        ImageDraw.Draw(pim).polygon(pcoors, fill = 255)
                else:
                    print("Simple Polygon Case")
                    # print(np.array(region_intersect[i].exterior.coords) / pow(2, level))
                    pcoors = np.array(np.array(region_intersect[i].exterior.coords) / pow(2, level)).astype(int)
                    print("Now drawing on image")
                    ImageDraw.Draw(pim).polygon(pcoors, fill=255)
        self.overlayim = pim
        self.overlayim.show()
        return self.overlayim
        print("Getting Overlay ", method)