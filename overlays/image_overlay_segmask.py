# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains class for overlaying segmentation maps"""

import openslide as ops
from PIL import Image

class SegMaskByPixel():
    def __init__(self, filename, wsiObj, bb_height, bb_width):
        """Initialises a SegMaskByPixel object

        Args:
            filename: WSI file containing the overlay (either TIFF, NDPI, JP2..)
            wsiObj: Whole-Slide Image to be overlayed
            bb_height: height of viewing window
            bb_width: width of viewing window
        """
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
                    self.nlevel = self.ovObj.level_count
                    self.leveldim = self.ovObj.level_dimensions
                except:
                    print("Undefined Behaviour or Unsupported File Format")
        else:
            self.ovObj = ops.ImageSlide(filename)
            self.type = "Image"
            print("Image has been read")
            self.nlevel = self.ovObj.level_count
            self.leveldim = self.ovObj.level_dimensions
        self.bb_height = bb_height
        self.bb_width = bb_width
        self.clevel, self.olevel = None, None
        self.hratio, self.wratio = None, None
        self.wsidim = [wsiObj.level_dimensions[i] for i in range(len(wsiObj.level_dimensions))]
        self.overlayim = None
        if self.type=="Image":
            for i in range(len(self.wsidim)):
                if self.wsidim[i] == self.leveldim[0]:
                    self.levelfetch = i
                    break
        else:
            self.levelfetch = self.nlevel - 1

    def get_overlay_simple(self, level, coorw, coorh, width, height):
        coor_low_w = pow(2, level) * coorw
        coor_low_h = pow(2, level) * coorh
        self.overlayim = self.ovObj.read_region((coor_low_w, coor_low_h), level, (width, height))
        return self.overlayim

    def get_overlay(self, level, coorw, coorh, width, height, method=None, step=None, class_states=None):
        """Fetches segmentation for the specified region, and level

        Args:
            level: Current level of self.wsiObj
            coorw: Width Coordinate of top left corner
            coorh: Height Coordinate of top left corner
            width: width of the overlay
            height: height of the overlay
            method: Operation performed {'zoom_in', 'zoom_out', 'pan', ..}
            step: Pan step

        Return:
            overlayim: Overlay image of size [width, height, 1]
        """
        if self.type=="OpenSlide":
            return self.get_overlay_openslide(level, coorw, coorh, width, height, method, step)
        elif self.type=="Image":
            print("Reading as Image format")
            return self.get_overlay_image(level, coorw, coorh, width, height, method, step)
        else:
            print("Unsupported File Format")
            return 1

    def get_overlay_image(self, level, coorw, coorh, width, height, method, step):
        self.coor_low_w = int(pow(2, level - self.levelfetch) * coorw)
        self.coor_low_h = int(pow(2, level - self.levelfetch) * coorh)
        self.cur_width = int(pow(2, level - self.levelfetch) * width)
        self.cur_height = int(pow(2, level - self.levelfetch) * height)
        self.low_width = int(pow(2, level - self.levelfetch) * width)
        self.low_height = int(pow(2, level - self.levelfetch) * height)
        print("Getting Image Overlay ", method)
        if method == "init":
            print("Inside init method")
            self.overlayim = self.ovObj.read_region((self.coor_low_w, self.coor_low_h), 0,
                                                    (self.cur_width, self.cur_height))
            self.overlayim = self.overlayim.resize((width, height))
            self.overlayim.show()
            return Image.eval(self.overlayim, lambda px: min(int(255*px/99), 255))
        elif method == "zoom_in":
            left = int(self.bb_width / 4)
            top = int(self.bb_height / 4)
            right = int(3 * self.bb_width / 4)
            bottom = int(3 * self.bb_height / 4)
            self.overlayim = self.overlayim.crop((left, top, right, bottom))
            self.overlayim = self.overlayim.resize((width, height))
            return Image.eval(self.overlayim, lambda px: min(int(255*px/99), 255))
        elif method == "zoom_out":
            pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
            self.overlayim = self.overlayim.resize((int(self.bb_width / 2), int(self.bb_height / 2)))
            pim.paste(self.overlayim, (int(self.bb_width / 4), int(self.bb_height / 4)))
            # pim.show()
            # Now read 4 surrounding regions
            tim_top = self.ovObj.read_region((self.coor_low_w, self.coor_low_h), 0,
                                             (self.cur_width, int(self.cur_height / 4)))
            tim_top = tim_top.resize((self.bb_width, int(self.bb_height / 4)))
            pim.paste(tim_top, (0, 0))

            tim_bottom = self.ovObj.read_region((self.coor_low_w, self.coor_low_h + int(3 * self.low_height / 4)),
                                                0,
                                                (self.cur_width, int(self.cur_height / 4)))
            tim_bottom = tim_bottom.resize((self.bb_width, int(self.bb_height / 4)))
            pim.paste(tim_bottom, (0, int(3 * self.bb_height / 4)))

            tim_left = self.ovObj.read_region((self.coor_low_w, self.coor_low_h + int(self.low_height / 4)),
                                              0,
                                              (int(self.cur_width / 4), int(self.cur_height / 2)))
            tim_left = tim_left.resize((int(self.bb_width / 4), int(self.bb_height / 2)))
            pim.paste(tim_left, (0, int(self.bb_height / 4)))

            tim_right = self.ovObj.read_region(
                (self.coor_low_w + int(3 * self.low_width / 4), self.coor_low_h + int(self.low_height / 4)),
                0, (int(self.cur_width / 4), int(self.cur_height / 2)))
            tim_right = tim_right.resize((int(self.bb_width / 4), int(self.bb_height / 2)))
            pim.paste(tim_right, (int(3 * self.bb_width / 4), int(self.bb_height / 4)))

            # self.ovObj.read_region((self.coor_low_w, self.coor_low_h), self.levelfetch,
            #                             (self.cur_width, self.cur_height)).show()
            self.overlayim = pim
            return Image.eval(self.overlayim, lambda px: min(int(255*px/99), 255))
        if step:
            if method == "left":
                print("Inside left method ", step)
                pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
                self.overlayim = self.overlayim.crop((0, 0, int((1 - step) * self.bb_width), self.bb_height))
                pim.paste(self.overlayim, (int(step * self.bb_width), 0))
                tim = self.ovObj.read_region((self.coor_low_w, self.coor_low_h), 0,
                                             (int(self.cur_width * step), self.cur_height))
                tim = tim.resize((int(step * self.bb_width), self.bb_height))
                pim.paste(tim, (0, 0))
                self.overlayim = pim
                return Image.eval(self.overlayim, lambda px: min(int(255*px/99), 255))
            elif method == "right":
                print("Inside right method ", step)
                pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
                self.overlayim = self.overlayim.crop((int(step * self.bb_width), 0, self.bb_width, self.bb_height))
                pim.paste(self.overlayim, (0, 0))
                tim = self.ovObj.read_region((self.coor_low_w + int((1 - step) * self.low_width), self.coor_low_h),
                                             0, (int(self.cur_width * step), self.cur_height))
                tim = tim.resize((int(step * self.bb_width), self.bb_height))
                pim.paste(tim, (int((1 - step) * self.bb_width), 0))
                self.overlayim = pim
                return Image.eval(self.overlayim, lambda px: min(int(255*px/99), 255))
            if method == "up":
                print("Inside left method ", step)
                pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
                self.overlayim = self.overlayim.crop((0, 0, self.bb_width, int((1 - step) * self.bb_height)))
                pim.paste(self.overlayim, (0, int(step * self.bb_height)))
                tim = self.ovObj.read_region((self.coor_low_w, self.coor_low_h), 0,
                                             (self.cur_width, int(step * self.cur_height)))
                tim = tim.resize((self.bb_width, int(step * self.bb_height)))
                pim.paste(tim, (0, 0))
                self.overlayim = pim
                return Image.eval(self.overlayim, lambda px: min(int(255*px/99), 255))
            if method == "down":
                print("Inside left method ", step)
                pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
                self.overlayim = self.overlayim.crop((0, int(step * self.bb_height), self.bb_width, self.bb_height))
                pim.paste(self.overlayim, (0, 0))
                tim = self.ovObj.read_region((self.coor_low_w, self.coor_low_h + int((1 - step) * self.low_height)),
                                             0,
                                             (self.cur_width, int(step * self.cur_height)))
                tim = tim.resize((self.bb_width, int(step * self.bb_height)))
                pim.paste(tim, (0, int((1 - step) * self.bb_height)))
                self.overlayim = pim
                return Image.eval(self.overlayim, lambda px: min(int(255*px/99), 255))

    def get_overlay_openslide(self, level, coorw, coorh, width, height, method=None, step=None):
        ## Currenly leaving out: Case in which a level0, level1 segmask are left out (missing alignment)
        ## Handled in image upwards
        if level < self.nlevel:
            print(self.wsidim[level])
            print(self.leveldim[level])
            if self.wsidim[level]==self.leveldim[level]:
                print("Inside Simple Overlay")
                return self.get_overlay_simple(level, coorw, coorh, width, height)
            # Write here as else condition for handling missing alignment cases

        self.coor_low_w = pow(2, level) * coorw
        self.coor_low_h = pow(2, level) * coorh
        self.cur_width = int(pow(2, level - self.levelfetch) * width)
        self.cur_height = int(pow(2, level - self.levelfetch) * height)
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
            # pim.show()

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