import glob
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, f1_score

pred_path = 'F:\\abhinav\\patches\\log_lstm_val_c1\\predictions'
label_path = 'F:\\abhinav\\patches\\lstm_data_label'

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

flist = glob.glob(pred_path + os.sep + '*.npy')
preds_all = []
labels_all = []
for i in flist:
    preds = np.load(i)
    preds = softmax(preds)[:,1]
    wsi_name = i.split(os.sep)[-1].split('_(')[0]
    labels = np.load(label_path + os.sep + wsi_name + os.sep + i.split(os.sep)[-1].replace('preds', 'label')).reshape(-1)
    labels = (labels > 0.5).astype(np.int)
    preds_all.extend(preds)
    labels_all.extend(labels)
    # print('Yes')

print(len(preds_all), len(labels_all))
preds_all = np.array(preds_all)
labels_all = np.array(labels_all)
# print(preds_all)
for i in range(0, 105, 5):
    preds = (preds_all>(i/100.)).astype(np.int)
    print(np.mean(preds==labels_all), np.sum(labels_all), np.sum(preds))
    print(precision_score(labels_all, preds), recall_score(labels_all, preds), f1_score(labels_all, preds))



