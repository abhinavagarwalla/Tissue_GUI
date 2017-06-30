# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains class for overlaying heatmaps"""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from interface.image_slide import ImageClass

class HeatMap():
    def __init__(self, filename, wsiObj, bb_height, bb_width):
        """Initialises a Heatmap object
        
        Args:
            filename: Folder containing heatmaps at various resolutions
            wsiObj: Whole-Slide Image to be overlayed
            bb_height: height of viewing window
            bb_width: width of viewing window
        """
        self.wsidim = [wsiObj.level_dimensions[i] for i in range(len(wsiObj.level_dimensions))]
        self.ovObj = ImageClass(filename)
        self.level_fetch = 0
        for i in range(len(self.wsidim)):
            if (self.wsidim[i] == self.ovObj.level_dimensions[0]).all():
                self.level_fetch = i
                continue

        self.type = "Image"
        self.overlayim = None
        self.cmap = plt.get_cmap("jet")

    def get_overlay(self, level, coorw, coorh, width, height, method=None, step=None, class_states=None):
        """Fetches heatmap overlay for the specified region, and level

        Args:
            level: Current level of self.wsiObj
            coorw: Width Coordinate of top left corner
            coorh: Height Coordinate of top left corner
            width: width of the overlay
            height: height of the overlay
        
        Return:
            overlayim: Overlay image of size [width, height, 1]
        """
        print("Getting Simple Overlay", self.level_fetch, level, coorw, coorh, width, height)
        if level >= self.level_fetch:
            self.overlayim = self.ovObj.read_region((coorw, coorh), level - self.level_fetch, (width, height))
        else:
            level_diff = level - self.level_fetch
            coor_low_w = int(pow(2, level_diff) * coorw)
            coor_low_h = int(pow(2, level_diff) * coorh)
            width_low = int(pow(2, level_diff) * width)
            height_low = int(pow(2, level_diff) * height)
            self.overlayim = self.ovObj.read_region((coor_low_w, coor_low_h), 0, (width_low, height_low))
            self.overlayim = self.overlayim.resize((width, height), Image.BILINEAR)

        self.overlayim = np.array(self.overlayim.convert("L")) / 99
        self.overlayim = np.uint8(self.cmap(self.overlayim) * 255)
        self.overlayim = Image.fromarray(self.overlayim)
        bands = list(self.overlayim.split())
        bands[3] = bands[3].point(lambda x: x * 0.2)
        self.overlayim = Image.merge(self.overlayim.mode, bands)
        return self.overlayim
