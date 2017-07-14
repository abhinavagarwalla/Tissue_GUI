import glob
import openslide as ops
import os
from PIL import Image
import numpy as np

base_path = 'F:\\abhinav\\patches\\lstm_data\\vis'
pred_list = glob.glob(base_path + os.sep + 'predictions\\*')
# print(pred_list)

wsi_obj = ops.OpenSlide('//shaban-pc/Camelyon16/Dataset/Original/Train/Tumor/Tumor_058.tif')
# overim = np.bool(wsi_obj.level_dimensions[0])
overim = Image.new('1', wsi_obj.level_dimensions[0], 0)
def f(pid):
    coors = pred_list[pid].split(os.sep)[-1].split('_(')[-1].split(')')[0]
    w, h = list(map(int, coors.split(',')))
    im = wsi_obj.read_region((w, h), 0, (1792, 1792))

    pim = Image.new('RGBA', im.size, color=(0, 0, 0, 127))
    preds = np.load(pred_list[pid])
    preds = (preds < 0.35).astype(np.int)
    print(pid,  pred_list[pid].split(os.sep)[-1], preds.shape, np.sum(preds))
    # print(preds)
    iter = 0
    for i in range(0, 224*8, 224):
        for j in range(0, 224*8, 224):
            # print(pid,  pred_list[pid].split(os.sep)[-1], preds[iter])
            if not preds[iter]:
                patchim = Image.new('RGBA', (224, 224), color=(64, 64, 64, 64))
                pim.paste(patchim, (i, j))
            iter += 1
    im = Image.alpha_composite(im, pim)
    im.save(base_path + os.sep + 'combination' + os.sep + pred_list[pid].split(os.sep)[-1].replace('.npy', '.png'))
    # exit()

if __name__=='__main__':
    from multiprocessing import Pool
    mp = Pool(16)
    mp.map(f, range(len(pred_list)))
    # f(1)