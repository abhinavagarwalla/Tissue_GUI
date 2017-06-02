import os

class Config():
    # MASK_PATH = None
    # WSI_PATH = None
    # CHECKPOINT_PATH = None
    MASK_PATH = '..\\Tissue_Data\\data\\Test_001_Mask.tif'
    WSI_PATH = '..\\Tissue_Data\\data\\Test_001.tif'
    CHECKPOINT_PATH = '..\\Tissue_Data\\data\\model.ckpt-35'
    TARGET_STAIN_PATH = 'resource' + os.sep + 'targetImage.jpg'
    RESULT_PATH = os.getcwd() + os.sep + 'results'
    LEVEL_UPGRADE = 5
    LEVEL_FETCH = 1
    PATCH_SIZE = 252 #124
    OUTPUT_SIZE = 164 #36
    DIFF_SIZE = PATCH_SIZE - OUTPUT_SIZE
    BATCH_SIZE = 64