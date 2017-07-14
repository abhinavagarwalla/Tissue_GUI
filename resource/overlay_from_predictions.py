import glob
import openslide as ops
import os
from PIL import Image
import numpy as np

base_path = 'F:\\abhinav\\patches\\lstm_data\\vis'
pred_list = glob.glob(base_path + os.sep + 'predictions\\*')
print(pred_list)

wsi_obj = ops.OpenSlide('//shaban-pc/Camelyon16/Dataset/Original/Train/Tumor/Tumor_058.tif')
overim = Image.new('1', wsi_obj.level_dimensions[4], 0)
def f(pid):
    coors = pred_list[pid].split(os.sep)[-1].split('_(')[-1].split(')')[0]
    w, h = list(map(int, coors.split(',')))
    w, h = int(w/16), int(h/16)
    # im = wsi_obj.read_region((w, h), 0, (1792, 1792))

    # pim = Image.new('RGBA', im.size, color=0)
    preds = np.load(pred_list[pid])
    preds = (preds < 0.35).astype(np.int)
    print(pid,  pred_list[pid].split(os.sep)[-1], preds.shape, np.sum(preds))
    # print(preds)
    iter = 0
    si = 14
    tum = 0
    for i in range(0, si*8, si):
        for j in range(0, si*8, si):
            # print(pid,  pred_list[pid].split(os.sep)[-1], preds[iter])
            if not preds[iter]:
                # print("YEs", w+i, h+j)
                # tum +=1
                patchim = Image.new('1', (si, si), color=1)
                overim.paste(patchim, (w+i, h+j))
            iter += 1
    # print("Value of tum= ", tum)
    # im = Image.alpha_composite(im, pim)
    # im.save(base_path + os.sep + 'combination' + os.sep + pred_list[pid].split(os.sep)[-1].replace('.npy', '.png'))
    # overim.paste()
    # exit()

if __name__=='__main__':
    # from multiprocessing import Pool
    # mp = Pool(16)
    # mp.map(f, range(len(pred_list)))
    for i in range(len(pred_list)):
        f(i)
        # print("Sum = ", np.sum(np.array(overim)))
    overim = overim.resize((int(overim.size[0]), int(overim.size[1])), Image.ANTIALIAS)
    overim.save(base_path + os.sep + 'overlay_058.png')
    overim = overim.resize((int(overim.size[0] / 4), int(overim.size[1] / 4)), Image.ANTIALIAS)
    overim.save(base_path + os.sep + 'overlay_058_d2.png')
    print("Done")