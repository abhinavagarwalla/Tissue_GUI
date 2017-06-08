import os

class Config():
    MASK_PATH = None
    WSI_PATH = None
    CHECKPOINT_PATH = None
    # MASK_PATH = 'C:\\Users\\abhinav\\Desktop\\Data\\data\\Test_005_Mask.tif'
    # WSI_PATH = 'C:\\Users\\abhinav\\Desktop\\Data\\data\\Test_005.tif'
    # CHECKPOINT_PATH = 'C:\\Users\\abhinav\\Desktop\\Data\\data\\model.ckpt-35'
    TARGET_STAIN_PATH = 'resource' + os.sep + 'targetImage.jpg'
    RESULT_PATH = os.getcwd() + os.sep + 'results'
    LEVEL_UPGRADE = None #5
    LEVEL_FETCH = None
    PATCH_SIZE = None #252 #124
    OUTPUT_SIZE = None #164 #36
    DIFF_SIZE = None #PATCH_SIZE - OUTPUT_SIZE
    BATCH_SIZE = 64

    STATE = None ## For setting either testing, training or patch creation

class PatchConfig():
    MASK_PATH = None
    WSI_FOLDER_PATH = '//shaban-pc/Camelyon16/Dataset/Original/Train' #None
    WSI_PATH = None
    # RESULT_PATH = os.getcwd() + os.sep + 'results'
    LEVEL_FETCH = 0 #None
    PATCH_SIZE = 384 #None  # 252 #124
    BATCH_SIZE = 128
    LEVEL_UPGRADE = 6
    RESULT_PATH = 'F:\\abhinav\\patches'

class TFRConfig():
    tfrecord_filename = "Camelyon_tfr" #None
    dataset_dir = "F:\\abhinav\\patches\\wsi" #None
    random_seed = 378
    num_shards = 10
    validation_size = 0.2
    COOR_PATH = 'F:\\abhinav\\patches\\Coors'

class TrainConfig():
    dataset_dir = 'F:\\abhinav\\patches\\wsi'
    log_dir = 'F:\\abhinav\\patches\\log'
    checkpoint_file = 'F:\\abhinav\\patches\\inception_resnet_v2_2016_08_30.ckpt'
    image_size = 299
    num_classes = 2
    labels_file = 'F:\\abhinav\\patches\\wsi\\labels.txt'
    file_pattern = 'Camelyon_tfr_%s_*.tfrecord'
    num_epochs = 5
    batch_size = 64
    initial_learning_rate = 0.001
    learning_rate_decay_factor = 0.9
    num_epochs_before_decay = 2
    source_size = 384

class TestConfig():
    pass