import glob
import openslide as ops
import os
from PIL import Image
import numpy as np

base_path = 'F:\\abhinav\\patches\\lstm_data\\vis'
save_path = 'F:\\abhinav\\patches\\lstm_data\\vis\\stacked'
pred_list = glob.glob(base_path + os.sep + 'combination_label\\*')
# print(pred_list)

wsi_obj = ops.OpenSlide('//shaban-pc/Camelyon16/Dataset/Original/Train/Tumor/Tumor_047.tif')
def f(pid):
    coors = pred_list[pid].split(os.sep)[-1].split('_(')[-1].split(')')[0]
    w, h = list(map(int, coors.split(',')))
    im = wsi_obj.read_region((w, h), 0, (1792, 1792))

    label = Image.open(pred_list[pid])
    lstm = Image.open(pred_list[pid].replace('label', 'lstm'))
    cnn = Image.open(pred_list[pid].replace('label', 'cnn'))

    pim = Image.new('RGBA', (im.size[0]*4, im.size[1]))
    pim.paste(im, (0,0))
    pim.paste(cnn, (im.size[0], 0))
    pim.paste(lstm, (2*im.size[0], 0))
    pim.paste(label, (3*im.size[0], 0))
    pim.save(save_path + os.sep + pred_list[pid].split(os.sep)[-1].replace('_preds.npy', '.png'))
    # exit()

if __name__=='__main__':
    from multiprocessing import Pool
    mp = Pool(16)
    mp.map(f, range(len(pred_list)))
    # f(1)