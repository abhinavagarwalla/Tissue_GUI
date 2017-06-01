import numpy as np
from PIL import Image
from scipy.io import loadmat
from shapely.geometry import Polygon, MultiPolygon
import cv2 as cv

class TumorRegion():
    def __init__(self, filename, wsiObj, bb_height, bb_width):
        self.ovObj = loadmat(filename)["predictions"]
        self.bb_height = bb_height
        self.bb_width = bb_width
        self.wsidim = [wsiObj.level_dimensions[i] for i in range(len(wsiObj.level_dimensions))]
        self.overlayim = None
        self.low_polygons = [Polygon(np.vstack((i[0][:,1],i[0][:,0])).T) for i in self.ovObj if i[0].shape[0]>3]
        self.npolygons = len(self.low_polygons)

    def get_overlay(self, level, coorw, coorh, width, height, method=None, step=None, class_states=None):
        print("Started Getting Overlay in Tumor Region")
        self.clevel = level-1
        self.coor_low_w = pow(2, self.clevel) * coorw
        self.coor_low_h = pow(2, self.clevel) * coorh
        self.low_width = pow(2, self.clevel) * width
        self.low_height = pow(2, self.clevel) * height
        imp = Polygon([(self.coor_low_w, self.coor_low_h), (self.coor_low_w + self.low_width, self.coor_low_h),
                       (self.coor_low_w + self.low_width, self.coor_low_h + self.low_height),
                       (self.coor_low_w, self.coor_low_h + self.low_height)])

        pim = np.zeros((height, width, 4), np.uint8)
        # pim *= 255

        for i in range(self.npolygons):
            if self.low_polygons[i].is_valid and imp.intersects(self.low_polygons[i]):
                region_intersect = imp.intersection(self.low_polygons[i])
                if region_intersect.is_valid:
                    if isinstance(region_intersect, MultiPolygon):
                        for kp in range(len(region_intersect)):
                            rel_img_coords = np.array(region_intersect[kp].exterior.coords)-(self.coor_low_w, self.coor_low_h)
                            pcoors = np.array(rel_img_coords/pow(2, self.clevel)).astype(np.int32).reshape((-1, 1, 2))
                            pim = cv.polylines(pim, [pcoors], True, (0, 255, 0, 255), 3)
                    elif isinstance(region_intersect, Polygon):
                        rel_img_coords = np.array(region_intersect.exterior.coords) - (self.coor_low_w, self.coor_low_h)
                        pcoors = np.array(rel_img_coords/pow(2, self.clevel)).astype(np.int32).reshape((-1, 1, 2))
                        pim = cv.polylines(pim, [pcoors], True, (0, 255, 0, 255), 3)
                    else:
                        pass
        self.overlayim = Image.fromarray(pim).convert("RGBA")
        # self.overlayim.show()
        return self.overlayim