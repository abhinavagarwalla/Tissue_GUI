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