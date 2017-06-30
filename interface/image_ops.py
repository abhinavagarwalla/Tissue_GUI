# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import cv2 as cv
import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
from interface.image_slide import ImageClass

from overlays.image_overlay_heatmap import HeatMap
from overlays.image_overlay_nuclei import NucleiPoints
from overlays.image_overlay_segmask import SegMaskByPixel
from overlays.image_overlay_tumor_region import TumorRegion


class DisplayImage():
    def __init__(self, filename, bb_height, bb_width):
        self.wsiObj = ImageClass(filename)
        self.bb_height = bb_height
        self.bb_width = bb_width
        self.level = self.wsiObj.level_count - 1
        self.leveldim = self.wsiObj.level_dimensions
        self.imheight, self.imwidth = None, None                ##width, height for the current view
        self.starth, self.startw = None, None                   ##Start h,m for the current view
        self.curim = None                                       ##Stores the current view of the image
        self.orim = None                                        ##Here the lowest level image stored and displayed
        self.coor_cur_h, self.coor_cur_w = None, None           ##Actual coordinates in the current view
        self.coor_low_h, self.coor_low_w = None, None           ##Actual coordinates in the lowest view
        self.overlayObj = dict()
        self.overlayim = dict()

    def read_first(self):
        self.curim = self.wsiObj.read_region((0,0), self.level, self.wsiObj.level_dimensions[self.level])
        self.orim = self.curim.copy()
        print(self.curim.size)

        self.imheight = self.curim.size[1]
        self.imwidth = self.curim.size[0]
        self.coor_cur_h, self.coor_cur_w = 0, 0
        self.coor_low_h, self.coor_low_w = 0, 0
        if self.imheight < self.bb_height or self.imwidth < self.bb_width:
            pim = self.pad_image()
            return ImageQt(self.orim), ImageQt(pim), self.level
        return ImageQt(self.orim), ImageQt(self.curim), self.level

    def get_image_in(self, factor=2):
        if self.imheight*factor < self.bb_height and self.imwidth*factor < self.bb_width:
            self.curim = self.curim.resize((2*self.curim.size[0], 2*self.curim.size[1]), Image.ANTIALIAS)
            self.imheight = self.curim.size[1]
            self.imwidth = self.curim.size[0]
            if self.imheight < self.bb_height or self.imwidth < self.bb_width:
                pim = self.pad_image()
                return ImageQt(pim)
        if self.imheight*factor >= self.bb_height:
            if self.imwidth*factor >= self.bb_width:
                print("Inside Popular condition")
                if self.level:
                    centerh = self.coor_cur_h + self.imheight / 2
                    centerw = self.coor_cur_w + self.imwidth / 2
                    left = int(centerw - self.bb_width / 4)
                    top = int(centerh - self.bb_height / 4)
                    self.level -= 1
                    self.coor_cur_h = 2 * top
                    self.coor_cur_w = 2 * left
                    self.coor_low_h = int(pow(2, self.level) * self.coor_cur_h)
                    self.coor_low_w = int(pow(2, self.level) * self.coor_cur_w)
                    # self.curim.show()
                    self.curim = self.wsiObj.read_region((self.coor_low_w, self.coor_low_h), self.level,
                                                         (self.bb_width, self.bb_height))
                    self.imheight = self.curim.size[1]
                    self.imwidth = self.curim.size[0]
                    ##Add padding Code here
                    if self.imheight < self.bb_height or self.imwidth < self.bb_width:
                        pim = self.pad_image()
                        return ImageQt(pim)
                    return ImageQt(self.curim)
                else:
                    return ImageQt(self.curim)
            else:
                print("I am in height condition")
                if self.level:
                    centerh = self.imheight / 2
                    left = 0
                    top = int(centerh - self.bb_height / 4)
                    self.level -= 1
                    self.coor_cur_h = 2 * top
                    self.coor_cur_w = 2 * left
                    self.coor_low_h = int(pow(2, self.level) * self.coor_cur_h)
                    self.coor_low_w = int(pow(2, self.level) * self.coor_cur_w)
                    self.curim = self.wsiObj.read_region((self.coor_low_w, self.coor_low_h), self.level,
                                                         (self.imwidth*2, self.bb_height))

                    self.imheight = self.curim.size[1]
                    self.imwidth = self.curim.size[0]
                    if self.imheight < self.bb_height or self.imwidth < self.bb_width:
                        pim = self.pad_image()
                        return ImageQt(pim)
                else:
                    return ImageQt(self.curim)
        if self.imwidth*factor >= self.bb_width:
            print("I am in width condition")
            print(self.imwidth, self.bb_width)
            if self.level:
                centerw = self.imwidth / 2
                left = centerw - int(self.bb_width / 4)
                top = 0
                self.level -= 1
                self.coor_cur_h = 2 * top
                self.coor_cur_w = 2 * left
                self.coor_low_h = int(pow(2, self.level) * self.coor_cur_h)
                self.coor_low_w = int(pow(2, self.level) * self.coor_cur_w)
                self.curim = self.wsiObj.read_region((self.coor_low_w, self.coor_low_h), self.level,
                                                     (self.bb_width, 2*self.im_height))

                self.imheight = self.curim.size[1]
                self.imwidth = self.curim.size[0]
                if self.imheight < self.bb_height or self.imwidth < self.bb_width:
                    pim = self.pad_image()
                    return ImageQt(pim)
            else:
                return ImageQt(self.curim)

    def get_image_out(self, factor=2):
        if self.imheight < self.bb_height and self.imwidth < self.bb_width:
            print("Inside the right box")
            self.curim = self.curim.resize((int(self.curim.size[0]/2), int(self.curim.size[1]/2)), Image.ANTIALIAS)
            self.imheight = self.curim.size[1]
            self.imwidth = self.curim.size[0]
            if self.imheight < self.bb_height or self.imwidth < self.bb_width:
                pim = self.pad_image()
                return ImageQt(pim)
        if self.imheight >= self.bb_height:
            if self.imwidth >= self.bb_width:
                print("Out: Inside Popular condition")
                if self.level!=self.wsiObj.level_count-1:
                    centerh = self.coor_cur_h + self.imheight / 2
                    centerw = self.coor_cur_w + self.imwidth / 2
                    self.level += 1
                    self.coor_cur_h = int(centerh/2 - self.bb_height/2)
                    self.coor_cur_w = int(centerw/2 - self.bb_width/2)
                    self.coor_low_h = int(pow(2, self.level) * self.coor_cur_h)
                    self.coor_low_w = int(pow(2, self.level) * self.coor_cur_w)
                    self.curim = self.wsiObj.read_region((self.coor_low_w, self.coor_low_h), self.level,
                                                         (self.bb_width, self.bb_height))
                    print("REgion Processing Complete")
                    self.imheight = self.curim.size[1]
                    self.imwidth = self.curim.size[0]
                    ## Add padding code here
                    if self.imheight < self.bb_height or self.imwidth < self.bb_width:
                        pim = self.pad_image()
                        return ImageQt(pim)
                    return ImageQt(self.curim)
                else:
                    return ImageQt(self.curim)
            else:
                print("I am in height condition")
                if self.level != self.wsiObj.level_count - 1:
                    centerh = self.coor_cur_h + self.imheight / 2
                    centerw = self.coor_cur_w + self.imwidth / 2
                    self.level += 1
                    self.coor_cur_h = int(centerh / 2 - self.bb_height / 2)
                    self.coor_cur_w = int(centerw / 2 - self.imwidth / 4)
                    self.coor_low_h = pow(2, self.level) * self.coor_cur_h
                    self.coor_low_w = pow(2, self.level) * self.coor_cur_w
                    self.curim = self.wsiObj.read_region((self.coor_low_w, self.coor_low_h), self.level,
                                                         (int(self.imwidth/2), self.bb_height))

                    self.imheight = self.curim.size[1]
                    self.imwidth = self.curim.size[0]
                    if self.imheight < self.bb_height or self.imwidth < self.bb_width:
                        pim = self.pad_image()
                        return ImageQt(pim)
                else:
                    return ImageQt(self.curim)
        if self.imwidth >= self.bb_width:
            print("I am in width condition")
            print(self.imwidth, self.bb_width)
            if self.level != self.wsiObj.level_count - 1:
                centerh = self.coor_cur_h + self.imheight / 2
                centerw = self.coor_cur_w + self.imwidth / 2
                self.level += 1
                self.coor_cur_h = int(centerh / 2 - self.imheight / 4)
                self.coor_cur_w = int(centerw / 2 - self.bb_width / 2)
                self.coor_low_h = pow(2, self.level) * self.coor_cur_h
                self.coor_low_w = pow(2, self.level) * self.coor_cur_w
                self.curim = self.wsiObj.read_region((self.coor_low_w, self.coor_low_h), self.level,
                                                     (self.bb_width, int(self.imheight/2)))

                self.imheight = self.curim.size[1]
                self.imwidth = self.curim.size[0]
                if self.imheight < self.bb_height or self.imwidth < self.bb_width:
                    pim = self.pad_image()
                    return ImageQt(pim)
            else:
                return ImageQt(self.curim)

    def check_boundaries(self, w, h, W, H, imw, imh):
        if w>W or w<0:
            return False
        if h>H or h<0:
            return False
        if w+imw>W:
            return False
        if h+imh>H:
            return False
        return True

    def pad_image(self, image=None):
        if image:
            pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
            self.startw = int((pim.size[0] - image.size[0]) / 2)
            self.starth = int((pim.size[1] - image.size[1]) / 2)
            pim.paste(image, (self.startw, self.starth))
            return pim
        pim = Image.new("RGBA", (self.bb_width, self.bb_height), (255, 255, 255, 0))
        self.startw = int((pim.size[0] - self.curim.size[0]) / 2)
        self.starth = int((pim.size[1] - self.curim.size[1]) / 2)
        pim.paste(self.curim, (self.startw, self.starth))
        return pim

    def pan(self, value_x=None, value_y=None):
        if_updated = False
        if value_x != None and value_y != None:
            if self.check_boundaries(self.coor_cur_w + value_x, self.coor_cur_h + value_y,
                                     self.leveldim[self.level][0], self.leveldim[self.level][1],
                                     self.bb_width, self.bb_height):
                    self.coor_cur_w = int(self.coor_cur_w + value_x)
                    self.coor_low_w = pow(2, self.level) * self.coor_cur_w

                    self.coor_cur_h = int(self.coor_cur_h + value_y)
                    self.coor_low_h = pow(2, self.level) * self.coor_cur_h
                    if_updated = True
        self.curim = self.wsiObj.read_region((self.coor_low_w, self.coor_low_h), self.level,
                                                 (self.bb_width, self.bb_height))
        self.imheight = self.curim.size[1]
        self.imwidth = self.curim.size[0]
        ## Pad if smaller region
        if self.imheight < self.bb_height or self.imwidth < self.bb_width:
            pim = self.pad_image()
            return ImageQt(pim), if_updated
        return ImageQt(self.curim), if_updated

    def read_first_overlay(self, filename, method=None, method_update="init", states=None):
        print(method)
        if method=="Segmentation Mask (by Pixel)":
            print("Inside Segmentation")
            self.overlayObj["Seg"] = SegMaskByPixel(filename, self.wsiObj, self.bb_height, self.bb_width)
            self.overlayim["Seg"] = self.overlayObj["Seg"].get_overlay(self.level, self.coor_cur_w, self.coor_cur_h, self.imwidth,
                                                         self.imheight, method_update)
            self.overlay_on_orig_image(state="Seg")
            return self.overlay_all(states)
        elif method=="Tumor Region":
            print("Tumor Regions")
            self.overlayObj["Reg"] = TumorRegion(filename, self.wsiObj, self.bb_height, self.bb_width)
            self.overlayim["Reg"] = self.overlayObj["Reg"].get_overlay(self.level, self.coor_cur_w, self.coor_cur_h, self.imwidth,
                                                         self.imheight, method_update)
            self.overlay_on_orig_image(state="Reg")
            return self.overlay_all(states)
        elif method=="Heatmap":
            print("HeatMap")
            self.overlayObj["Heat"] = HeatMap(filename, self.wsiObj, self.bb_height, self.bb_width)
            self.overlayim["Heat"] = self.overlayObj["Heat"].get_overlay(self.level, self.coor_cur_w, self.coor_cur_h,
                                                                       self.imwidth,
                                                                       self.imheight, method_update)
            self.overlay_on_orig_image(state="Heat")
            return self.overlay_all(states)
        elif method=="Nuclei Position":
            print("Nuclei Position")
            self.overlayObj["Nuclei"] = NucleiPoints(filename, self.wsiObj, self.bb_height, self.bb_width)
            self.overlayim["Nuclei"] = self.overlayObj["Nuclei"].get_overlay(self.level, self.coor_cur_w, self.coor_cur_h, self.imwidth,
                                                         self.imheight, method_update)
            self.overlay_on_orig_image(state="Nuclei")
            return self.overlay_all(states)

    def update_overlay(self, method_update="init", step=None, states=None, ov_no_update=None, class_states=None):
        print("inside update_overlay in ImageOps ", states)
        if ov_no_update:
            return self.overlay_all(states)
        for k, v in states.items():
            if v:
                print("Value of class statues: ", class_states)
                self.overlayim[k] = self.overlayObj[k].get_overlay(self.level, self.coor_cur_w, self.coor_cur_h, self.imwidth,
                                                         self.imheight, method_update, step, class_states=class_states)
        return self.overlay_all(states)

    def overlay_all(self, states):
        print("Value of states: ", states)
        self.t = self.curim.copy()
        for k, v in states.items():
            if v:
                print(k, v)
                if k=="Seg":
                    print("Blending Image together", self.t.size, self.overlayim["Seg"].size)
                    self.t = Image.blend(self.t, self.overlayim["Seg"], 0.7)
                elif k=="Reg":
                    # self.t = ImageChops.multiply(self.t, self.overlayim["Reg"])
                    self.t = Image.alpha_composite(self.t, self.overlayim["Reg"])
                elif k=="Heat":
                    # self.t = Image.blend(self.t, self.overlayim["Heat"], 0.4)
                    self.t = Image.alpha_composite(self.t, self.overlayim["Heat"])
                elif k=="Nuclei":
                    self.t = Image.alpha_composite(self.t, self.overlayim["Nuclei"])
        if self.imheight < self.bb_height or self.imwidth < self.bb_width:
            pim = self.pad_image(image=self.t)
            # pim.show()
            return ImageQt(pim)
        return ImageQt(self.t)

    def overlay_on_orig_image(self, state=None):
        if state=="Seg":
            # self.overlayim["Seg"].save("Segmentation_overlay.png")
            self.overlayim["Seg"] = Image.blend(self.curim, self.overlayim["Seg"], 0.7)
        elif state=="Reg":
            # self.overlayim["Reg"].save("Region_overlay.png")
            self.overlayim["Reg"] = Image.alpha_composite(self.curim, self.overlayim["Reg"])
        elif state=="Heat":
            # self.overlayim["Heat"].save("Heatmap_overlay.png")
            # self.overlayim["Heat"] = Image.blend(self.curim, self.overlayim["Heat"], 0.4)
            self.overlayim["Heat"] = Image.alpha_composite(self.curim, self.overlayim["Heat"])
        elif state=="Nuclei":
            print("overlaying on original image", self.curim, self.overlayim["Nuclei"])
            self.overlayim["Nuclei"] = Image.alpha_composite(self.curim, self.overlayim["Nuclei"])
        # self.curim.save("Current_Image.png")

    def get_number_classes(self):
        if self.overlayim["Nuclei"]:
            return self.overlayObj["Nuclei"].nclasses, self.overlayObj["Nuclei"].class_names

    def get_info(self):
        ocvim = cv.cvtColor(np.array(self.orim), cv.COLOR_RGB2BGR)
        left = int(pow(2, self.level-len(self.leveldim)+1) * self.coor_cur_w)
        top = int(pow(2, self.level-len(self.leveldim)+1) * self.coor_cur_h)
        width = int(pow(2, self.level-len(self.leveldim)+1) * self.imwidth)
        height = int(pow(2, self.level-len(self.leveldim)+1) * self.imheight)
        cv.rectangle(ocvim, (left, top), (left+width, top+height),(0,0,0),1)
        ocvim = cv.cvtColor(np.array(ocvim), cv.COLOR_BGR2RGB)
        return ImageQt(Image.fromarray(ocvim).convert("RGBA"))

    def random_seek(self, w, h, isize):
        if int(w) > self.leveldim[-1][0]:
            return ImageQt(self.curim)
        if int(h - (isize.height()-self.leveldim[-1][1])/2) > self.leveldim[-1][1]:
            return ImageQt(self.curim)
        width = int(pow(2, self.level - len(self.leveldim) + 1) * self.imwidth)
        height = int(pow(2, self.level - len(self.leveldim) + 1) * self.imheight)
        self.coor_cur_w = pow(2, len(self.leveldim) - 1 - self.level) * int(w - (width/2))
        self.coor_cur_h = pow(2, len(self.leveldim) - 1 - self.level) * \
                          int(h - (height/2) - (isize.height()-self.leveldim[-1][1])/2)
        self.coor_cur_w = 0 if self.coor_cur_w < 0 else self.coor_cur_w
        self.coor_cur_h = 0 if self.coor_cur_h < 0 else self.coor_cur_h
        self.coor_low_w = pow(2, self.level) * self.coor_cur_w
        self.coor_low_h = pow(2, self.level) * self.coor_cur_h
        self.curim = self.wsiObj.read_region((self.coor_low_w, self.coor_low_h), self.level,
                                             (self.bb_width, self.bb_height))
        self.imheight = self.curim.size[1]
        self.imwidth = self.curim.size[0]
        ##Add padding here in case of smaller region fetched
        if self.imheight < self.bb_height or self.imwidth < self.bb_width:
            pim = self.pad_image()
            return ImageQt(pim)
        return ImageQt(self.curim)

    def get_current_coordinates(self):
        return self.coor_cur_w, self.coor_cur_h