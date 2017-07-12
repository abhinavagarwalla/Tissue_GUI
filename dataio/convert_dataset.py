# Copyright 2017 Abhinav Agarwalla. All Rights Reserved.
# Contact: agarwallaabhinav@gmail.com, abhinavagarwalla@iitkgp.ac.in
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts Image directory into TFRecords"""

import glob
import os
import random

from PyQt5.QtCore import QObject, pyqtSignal

from dataio.dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset
from dl_interface.model_config import TFRConfig

class TFRecordConverter(QObject):
    finished = pyqtSignal()

    def initialize(self):
        pass
        # if os.path.exists(PatchConfig.RESULT_PATH):
        #     shutil.rmtree(PatchConfig.RESULT_PATH)
        # os.mkdir(PatchConfig.RESULT_PATH)
        # os.mkdir(PatchConfig.RESULT_PATH + os.sep + "Coors")
        # os.mkdir(PatchConfig.RESULT_PATH + os.sep + "Ambiguous")
        # [os.mkdir(PatchConfig.RESULT_PATH + os.sep + i) for i in self.classes]

    def run(self):
        """Splits dataset into training and validation set at WSI level.
        Saves data as TFRecords with training and validation prefix"""
        self.initialize()
        if not TFRConfig.tfrecord_filename:
            raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')

        if not TFRConfig.dataset_dir:
            raise ValueError('dataset_dir is empty. Please state a dataset_dir argument.')

        if _dataset_exists(dataset_dir=TFRConfig.dataset_dir, _NUM_SHARDS=TFRConfig.num_shards,
                           output_filename=TFRConfig.tfrecord_filename):
            print('Dataset files already exist. Exiting without re-creating them.')
            return None

        if not TFRConfig.COOR_PATH:
            print("Coordinates files not found")

        # Get a list of photo_filenames like ['123.jpg', '456.jpg'...] and a list of sorted class names from parsing the subdirectories.
        photo_filenames, class_names = _get_filenames_and_classes(TFRConfig.dataset_dir)

        # Refer each of the class name to a specific integer number for predictions later
        class_names_to_ids = dict(zip(class_names, range(len(class_names))))

        # Find the number of validation examples we need
        coor_list = glob.glob(TFRConfig.COOR_PATH + os.sep + "*_coors.npy")
        num_validation = int(TFRConfig.validation_size * len(coor_list))

        # Divide the training datasets into train and test:
        random.seed(TFRConfig.random_seed)
        random.shuffle(coor_list)

        trainWSI = coor_list[num_validation:]
        validWSI = coor_list[:num_validation]

        trainWSI = [i.split(os.sep)[-1].split('_coors')[0] for i in trainWSI]
        validWSI = [i.split(os.sep)[-1].split('_coors')[0] for i in validWSI]

        training_filenames = [i for i in photo_filenames if i.split(')_')[-1].split('.')[0] in trainWSI]
        validation_filenames = [i for i in photo_filenames if i.split(')_')[-1].split('.')[0] in validWSI]

        random.shuffle(training_filenames)
        random.shuffle(validation_filenames)

        # First, convert the training and validation sets.
        _convert_dataset('train', training_filenames, class_names_to_ids,
                         dataset_dir=TFRConfig.dataset_dir, tfrecord_filename=TFRConfig.tfrecord_filename,
                         _NUM_SHARDS=TFRConfig.num_shards)
        _convert_dataset('validation', validation_filenames, class_names_to_ids,
                         dataset_dir=TFRConfig.dataset_dir, tfrecord_filename=TFRConfig.tfrecord_filename,
                         _NUM_SHARDS=TFRConfig.num_shards)

        # Finally, write the labels file:
        labels_to_class_names = dict(zip(range(len(class_names)), class_names))
        write_label_file(labels_to_class_names, TFRConfig.dataset_dir)

        print('\nFinished converting the %s dataset!' % (TFRConfig.tfrecord_filename))