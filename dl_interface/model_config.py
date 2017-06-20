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
    num_shards = 20
    validation_size = 0.2
    COOR_PATH = 'F:\\abhinav\\patches\\Coors'

class TrainConfig():
    dataset_dir = 'F:\\abhinav\\patches\\wsi'
    log_dir = 'F:\\abhinav\\patches\\log'
    checkpoint_file = 'F:\\abhinav\\patches\\inception_resnet_v2_2016_08_30.ckpt'
    image_size = 224
    num_classes = 2
    labels_file = 'F:\\abhinav\\patches\\wsi\\labels.txt'
    file_pattern = 'Camelyon_tfr_%s_*.tfrecord'
    num_epochs = 5
    batch_size = 16
    initial_learning_rate = 0.0002
    learning_rate_decay_factor = 0.9
    num_epochs_before_decay = 2
    source_size = 384

class ValidConfig():
    dataset_dir = 'F:\\abhinav\\patches\\wsi'
    log_dir = 'F:\\abhinav\\patches\\log'
    log_eval = 'F:\\abhinav\\patches\\log_eval'
    checkpoint_file = 'F:\\abhinav\\patches\\log\\model.ckpt-6001'
    image_size = 224
    num_classes = 2
    labels_file = 'F:\\abhinav\\patches\\wsi\\labels.txt'
    file_pattern = 'Camelyon_tfr_%s_*.tfrecord'
    num_epochs = 1
    batch_size = 16
    source_size = 384

class TestConfig():
    pass

class LSTMDataConfig():
    WSI_FOLDER_PATH = '//shaban-pc/Camelyon16/Dataset/Original/Train'  # None
    MASK_PATH = None #'C:\\Users\\abhinav\\Desktop\\Data\\data\\Test_005_Mask.tif'
    WSI_PATH = None #'C:\\Users\\abhinav\\Desktop\\Data\\data\\Test_005.tif'
    CHECKPOINT_PATH = 'F:\\abhinav\\patches\\log\\model.ckpt-6001'
    TARGET_STAIN_PATH = 'resource' + os.sep + 'targetImage.jpg'
    RESULT_PATH = 'F:\\abhinav\\patches\\lstm_data' #os.getcwd() + os.sep + 'results'
    LABEL_PATH = 'F:\\abhinav\\patches\\lstm_data_label'
    LEVEL_UPGRADE = 6
    LEVEL_FETCH = 0
    CONTEXT_DEPTH = 8
    PATCH_SIZE = 224  # 252 #124
    STRIDE = 0.5
    OUTPUT_SIZE = None  # 164 #36
    DIFF_SIZE = None  # PATCH_SIZE - OUTPUT_SIZE
    BATCH_SIZE = 512
    NUM_CLASSES = 2

class LSTMTrainConfig():
    DATA_IMAGES_PATH = 'F:\\abhinav\\patches\\lstm_data'
    DATA_LABELS_PATH = 'F:\\abhinav\\patches\\lstm_data_label'
    PATCH_SIZE = 8
    CHANNELS = 4096
    HIDDEN_SIZE = 512
    log_dir = 'F:\\abhinav\\patches\\log_lstm'
    batch_size = 10
    num_epochs = None
    checkpoint_file = None
    initial_learning_rate = 0.0001
    learning_rate_decay_factor = 0.5
    num_epochs_before_decay = 2