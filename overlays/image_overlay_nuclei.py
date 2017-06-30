# Copyright 2016 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat

class NucleiPoints():
    def __init__(self, filename, wsiObj, bb_height, bb_width):
        self.ovObj = loadmat(filename)["predictions"]
        self.bb_height = bb_height
        self.bb_width = bb_width
        self.wsidim = [wsiObj.level_dimensions[i] for i in range(len(wsiObj.level_dimensions))]
        self.overlayim = None
        self.preds_class = self.ovObj[:,2]
        self.nclasses = np.max(self.preds_class)

        self.class_names = None
        tmpc = open(filename.split('.')[0] + '_labels.txt').readlines()
        self.class_names = [i.split(',')[1].strip() for i in tmpc]
        self.npoints = len(self.preds_class)
        self.colors = [(255,0,255,255), (255,0,0,255), (0,255,0,255),
                       (255,128,0,255), (0,0,0,255), (0,0,255,255)]
        self.clsdict = {0: False, 1: False, 2: False, 3: False, 4: False, 5: False}

    def get_overlay(self, level, coorw, coorh, width, height, method=None, step=None, class_states=None):
        print("Started Getting Overlay inside Nuclei position")
        self.clevel = level-1
        self.coor_low_w = pow(2, self.clevel) * (coorw)
        self.coor_low_h = pow(2, self.clevel) * (coorh)
        self.low_width = pow(2, self.clevel) * (width)
        self.low_height = pow(2, self.clevel) * (height)
        imp = [(self.coor_low_w, self.coor_low_h),
               (self.coor_low_w + self.low_width, self.coor_low_h + self.low_height)]

        points = self.ovObj[(self.ovObj[:,0] > imp[0][1]) & (self.ovObj[:,0] < imp[1][1]) &
                            (self.ovObj[:,1] > imp[0][0]) & (self.ovObj[:,1] < imp[1][0])]
        pointsW = ((points[:,1]-self.coor_low_w)/pow(2, self.clevel)).astype(np.int16)
        pointsH = ((points[:,0]-self.coor_low_h)/pow(2, self.clevel)).astype(np.int16)
        pim = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        d = ImageDraw.Draw(pim)
        if class_states:
            print(class_states)
            self.clsdict = class_states
        for i in range(self.nclasses):
            if self.clsdict[i]:
                coors = list(zip(pointsW[points[:,2]==(i+1)], pointsH[points[:,2]==(i+1)]))
                d.point(coors, self.colors[i])
                if level<=4 and len(coors)!=0:
                    coor_array = np.array(coors)
                    for j in range(len(coor_array)):
                        d.ellipse([tuple(coor_array[j] - (2, 2)), tuple(coor_array[j] + (2, 2))], fill=self.colors[i])
        self.overlayim = pim
        return self.overlayim