# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fetches data batches for CNN training by using Tissue Mask"""

from itertools import product

import cv2
import numpy as np
from scipy import ndimage

from dl_interface.model_config import *
from interface.image_slide import ImageClass
from preprocessing import preprocessing_factory

class Data():
    """Reads WSI region according to coordinates extracted from Tissue Mask
    """
    def __init__(self, preprocessor, outshape):
        """Data Class Constructor
        Args:
            preprocessor: Name of preprocessing function to be used
            outshape: shape of output tensor
        """
        self.wsi = ImageClass(Config.WSI_PATH)
        self.output_size = int(outshape[1])
        Config.DIFF_SIZE = Config.PATCH_SIZE - self.output_size
        self.preprocessor = preprocessing_factory.get_preprocessing_fn(name=preprocessor)
        self.coors = self.get_coordinates()
        self.nsamples = len(self.coors)
        self.iter = 0
        self.continue_flag = True
        self.data_completed = False
        self.write_to_folder_flag = True

    def delete_inside(self, boxes):
        """Delete coordinate box inside another box, i.e. remove duplicity in coordinates
        """
        boxes = np.array(boxes)
        boxes_new = []
        for i in range(len(boxes)):
            a = boxes[(boxes[:, 0] < boxes[i, 0]) & (boxes[:, 1] < boxes[i, 1]) &
                      ((boxes[:, 0] + boxes[:, 2]) > (boxes[i, 0] + boxes[i, 2])) &
                      ((boxes[:, 1] + boxes[:, 3]) > (boxes[i, 1] + boxes[i, 3]))]
            if len(a):
                print(len(a), a, boxes[i])
            else:
                boxes_new.append(boxes[i])
        return np.array(boxes_new)

    def get_coordinates(self):
        """Gets coordinates from Tissue Mask.
        After finding contours, it rejects very small contours and contruct bounding boxes for remaining
        """
        img = ndimage.imread(Config.MASK_PATH)
        contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        a = np.array([cv2.contourArea(i) for i in contours[1]])
        b = np.array(contours[1])
        order = a.argsort()[::-1]
        a, b = a[order], b[order]
        threshArea, threshPoints = 200, 10
        boxes = [cv2.boundingRect(i) for i in b[a > threshArea] if len(i) > threshPoints]
        # shapely_box = [box(i[1], i[0], i[1]+i[3], i[0]+i[2]) for i in boxes]
        boxes = self.delete_inside(boxes)
        # print(boxes)

        boxes = boxes * pow(2, Config.LEVEL_UPGRADE)
        print(boxes)
        coors = []
        ## Make it more complete
        for i in range(len(boxes)):
            a = range(max(0, boxes[i, 0] - Config.DIFF_SIZE),
                      min(self.wsi.level_dimensions[Config.LEVEL_FETCH][0],
                          boxes[i, 0] + boxes[i, 2] + Config.DIFF_SIZE), self.output_size)
            b = range(max(0, boxes[i, 1] - Config.DIFF_SIZE),
                      min(self.wsi.level_dimensions[Config.LEVEL_FETCH][1],
                          boxes[i, 1] + boxes[i, 3] + Config.DIFF_SIZE), self.output_size)
            coors.extend(list(product(a, b)))
        return coors

    def get_image_from_coor(self):
        """Fetches image region from WSI with the coordinates extracted by get_coordinates function
        """
        image_batch = []
        coor_batch = []
        while len(coor_batch) != Config.BATCH_SIZE:
            # re = random.randint(0, self.nsamples)
            im = np.array(self.wsi.read_region((pow(2, Config.LEVEL_FETCH) * self.coors[self.iter][0],
                                                pow(2, Config.LEVEL_FETCH) * self.coors[self.iter][1]),
                                               Config.LEVEL_FETCH,
                                               (Config.PATCH_SIZE, Config.PATCH_SIZE)).convert('RGB'))
            if np.mean(im) <= 240:
                im = self.preprocessor.preprocess_image(im)
                image_batch.append(im[:, :, ::-1])  # RGB to BGR
                coor_batch.append(self.coors[self.iter])
            self.iter += 1
            if self.iter == self.nsamples:
                self.continue_flag = False
                self.data_completed = True
                break

        image_batch = np.array(image_batch).reshape(-1, Config.PATCH_SIZE, Config.PATCH_SIZE, 3)
        return image_batch, coor_batch

    def save_predictions(self, preds, coors_batch, images=None):
        """Saves predictions as images
        Args:
            preds: Probabilites from network
            coors_batch: Coordinates for each position at WSI level
            images: secondary images to be save, if any
        """
        preds = (np.array(preds) * 100).astype(np.uint8)
        for i in range(Config.BATCH_SIZE):
            if self.write_to_folder_flag:
                cv2.imwrite(Config.RESULT_PATH + os.sep + str(coors_batch[i]) + "_tumor.png", preds[i, :, :, 0])
                cv2.imwrite(Config.RESULT_PATH + os.sep + str(coors_batch[i]) + "_non_tumor.png", preds[i, :, :, 1])
                # orim = Image.fromarray(images[i])
                # orim.save('results\\' + str(coors_batch[i]) + "_orig.png")