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
    pass

class TestConfig():
    pass