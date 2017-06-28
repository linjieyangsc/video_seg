import os
import sys
import cv2
from PIL import Image
import numpy as np
src_path = '/raid/ljyang/data/DAVIS/JPEGImages/480p'
pred_path = 'DAVIS/Results/Segmentations/480p/OSVOS'
sav_path = 'DAVIS/Results/Segmentations/480p/OSVOS_test-dev'
listFile = '/raid/ljyang/data/DAVIS/ImageSets/2017/test-dev.txt'
PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128]
with open(listFile, 'r') as f:
    fds = [line.strip() for line in f]
im_num = 0
iou =[] 
seq_n = 0
subfd_names = []
for i, fd in enumerate(fds):
    print fd
    file_list = os.listdir(os.path.join(src_path, fd))
    
    im_list = [name for name in file_list]
    im_list = sorted(im_list)
    pred_list = os.listdir(os.path.join(pred_path, fd))
    sub_fds = [name for name in pred_list if len(name) < 4]
    sub_fds = sorted(sub_fds)
    saveDir = os.path.join(sav_path, fd)
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    for im_name in im_list[1:-1]:
        iou_im = 0
        scores = []
        for j, sub_fd in enumerate(sub_fds):

            score = np.load(os.path.join(pred_path, fd, sub_fd, im_name[:-4] + '.npy'))
            scores.append(score)
        im_size = scores[0].shape
        bg_score = np.ones(im_size) * 0.5
        scores = [bg_score] + scores
        score_all = np.stack(tuple(scores), axis = -1)
        label_pred = score_all.argmax(axis=2)
        result_path = os.path.join(saveDir, im_name[:-4] + '.png')
        res_im = Image.fromarray(label_pred.astype(np.uint8), mode="P")
        res_im.putpalette(PALETTE)
        res_im.save(result_path)
        print 'Saving ' + result_path
