# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Portion of code borrowed from: Shan, Shaban Camelyon'16 Attempt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Provides utilities for stain normalisation of patches."""

import cv2
import numpy as np

from dl_interface.model_config import *


class StrainNormalisation():
    def __init__(self):
        self.targetImg = cv2.imread(Config.TARGET_STAIN_PATH)
        self.targetImg = cv2.cvtColor(self.targetImg, 44);

        # Mean and Standard Deviation of Target image channels in Lab Colourspace
        t1, t2, t3 = cv2.split(self.targetImg)
        t1 = t1 / 2.55
        t2 = t2 + (-128)
        t3 = t3 + (-128)
        self.mT1, self.sdT1 = cv2.meanStdDev(t1)
        self.mT2, self.sdT2 = cv2.meanStdDev(t2)
        self.mT3, self.sdT3 = cv2.meanStdDev(t3)

    def preprocess_image(self, patch):
        """Pre-process one image for training or evaluation.

        Args:
          patch: 3-D numpy array [height, width, channels] with the image.

        Returns:
          3-D numpy array containing an appropriately scaled image
        """
        patch = cv2.cvtColor(patch, 44)

        # Mean and Standard Deviation of Source image channels in Lab Colourspace
        s1, s2, s3 = cv2.split(patch)
        s1 = s1 / 2.55
        s2 = s2 + (-128)
        s3 = s3 + (-128)
        mS1, sdS1 = cv2.meanStdDev(s1)
        mS2, sdS2 = cv2.meanStdDev(s2)
        mS3, sdS3 = cv2.meanStdDev(s3)

        if sdS1 == 0:
            sdS1 = 1;
        if sdS2 == 0:
            sdS2 = 1;
        if sdS3 == 0:
            sdS3 = 1;

        normLab_1 = ((s1 - mS1) * (self.sdT1 / sdS1)) + self.mT1;
        normLab_2 = ((s2 - mS2) * (self.sdT2 / sdS2)) + self.mT2;
        normLab_3 = ((s3 - mS3) * (self.sdT3 / sdS3)) + self.mT3;

        normLab = cv2.merge((normLab_1, normLab_2, normLab_3))
        normLab = np.float32(normLab)
        norm = cv2.cvtColor(normLab, 56)
        return norm

    def preprocess_batch(self, patches):
        res = []
        for i in patches:
            res.append(self.preprocess_image(i))
        return np.array(res)