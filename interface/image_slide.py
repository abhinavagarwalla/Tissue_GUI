# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains wrapper for reading different image formats (TIFF, JP2, PNG_Folder) """

import numpy as np
import openslide as ops
from PIL import Image
import matlab.engine as me
import matlab
import os

class ImageClass():
    def __init__(self, filename):
        """Initiates the appropriate format depending on file name
        
        Args:
            filename: Complete path to file, along with format
        
        Returns:
            ImageClass object
        """
        print("Initiating class with ", filename)
        olist = ["tif", "ndpi", "vms", "vmu", "svs", "mrxs", "scn", "svslide", "bif"]
        if filename.split('.')[-1] in olist:
            self.wsiObj = ops.OpenSlide(filename)
            self.level_count = self.wsiObj.level_count
            self.level_dimensions = self.wsiObj.level_dimensions
            self.type = "tiff"
        if ".jp2" in filename:
            self.filename = filename
            self.eng = me.start_matlab()
            self.eng.cd(os.getcwd() + os.sep + 'interface', nargout=0)
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
            print(self.tlist)
            self.level_count = len(self.tlist)
            self.level_dimensions = []
            for i in range(self.level_count):
                self.wsiObj.append(ops.ImageSlide(filename + "/" + self.tlist[i]))
                self.wsiObj[i].read_region((0, 0), 0, (10, 10))
                self.level_dimensions.append(self.wsiObj[i].level_dimensions[0])
            self.level_dimensions = np.array(self.level_dimensions)
            inds = self.level_dimensions[:,0].argsort()[::-1]
            self.level_dimensions = list(self.level_dimensions[inds])
            self.wsiObj = list(np.array(self.wsiObj)[inds])
            self.type = "png_folder"
            print(self.level_count, self.level_dimensions, type(self.level_dimensions), type(self.wsiObj), len(self.wsiObj))

    def read_region(self, coor_low, level, dim):
        """Reads the specified region
        
        Args:
            coor_low: Coordinates (w, h) at the lowest level (highest resolution)
            level: level to be fetched
            dim: dimensions of the region to be fetched
        
        Returns:
            RGBA Image of size dim
        """
        # print("Reading Region: ", coor_low, level, dim)
        if self.type=="tiff":
            return self.wsiObj.read_region(coor_low, level, dim)
        elif self.type=="jp2":
            coor_cur_w = int(coor_low[0] / pow(2, level))
            coor_cur_h = int(coor_low[1] / pow(2, level))
            hleft = max(1, coor_cur_h+1)
            hdown = min(self.level_dimensions[level][1], hleft + dim[1])
            wleft = max(1, coor_cur_w+1)
            wright = min(self.level_dimensions[level][0], wleft + dim[0])
            mim = self.eng.read_region(self.filename, level, matlab.int32([hleft, hdown, wleft, wright]))
            im = np.array(mim._data).reshape(mim.size, order='F')
            # print("Inside 1st condition", Image.fromarray(im).convert("RGBA").size)
            return Image.fromarray(im).convert("RGBA")
        elif self.type=="png_folder":
            print("Reading Region")
            return self.wsiObj[level].read_region(coor_low, 0, dim)
