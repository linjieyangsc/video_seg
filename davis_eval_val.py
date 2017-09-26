import os
import sys
import cv2
from PIL import Image
import numpy as np
def calcIoU(gt, pred, obj_n):
    assert(gt.shape == pred.shape)
    ious = np.zeros((obj_n), dtype=np.float32)
    for obj_id in range(1, obj_n+1):
        gt_mask = gt == obj_id
        pred_mask = pred == obj_id
        inter = gt_mask & pred_mask
    
        union = gt_mask | pred_mask
        if union.sum() == 0:
            ious[obj_id-1] = 1
        else:
            ious[obj_id-1] = float(inter.sum()) / union.sum()
    return ious
gt_path = '/raid/ljyang/data/DAVIS/Annotations/480p'
pred_path = sys.argv[1] #'DAVIS/Results/Segmentations/480p/OSVOS'
sav_path = 'DAVIS/Visualize'
listFile = '/raid/ljyang/data/DAVIS/ImageSets/2017/val.txt'
with open(listFile, 'r') as f:
    fds = [line.strip() for line in f]
im_num = 0
iou =[] 
seq_n = 0
subfd_names = []
for i, fd in enumerate(fds):
    print fd
    file_list = os.listdir(os.path.join(gt_path, fd))
    im_list = [name for name in file_list if len(name) > 4 and name[-4:]=='.png']
    im_list = sorted(im_list)
    pred_list = os.listdir(os.path.join(pred_path, fd))
    sub_fds = [name for name in pred_list if len(name) < 4]
    sub_fds = sorted(sub_fds)
    print sub_fds
    for sub_fd in sub_fds:
        subfd_names.append(fd+'/'+sub_fd)
    iou_seq = []
    for im_name in im_list[1:-1]:
        iou_im = 0
        scores = []
        label_gt = np.array(Image.open(os.path.join(gt_path, fd, im_name)))
        for j, sub_fd in enumerate(sub_fds):

            score = np.load(os.path.join(pred_path, fd, sub_fd, im_name[:-4] + '.npy'))
            scores.append(score)
        im_size = scores[0].shape
        bg_score = np.ones(im_size) * 0.5
        scores = [bg_score] + scores
        score_all = np.stack(tuple(scores), axis = -1)
        class_n = score_all.shape[2] - 1
        label_pred = score_all.argmax(axis=2)
        label_pred = np.array(Image.fromarray(label_pred.astype(np.uint8)).resize((label_gt.shape[1], label_gt.shape[0]), 
            Image.NEAREST))
        #cv2.resize(label_pred, label_gt.shape, label_pred, 0, 0, cv2.INTER_NEAREST)
        iou_seq.append(calcIoU(label_gt, label_pred, class_n))
    iou_seq = np.stack(iou_seq, axis=1)
    print iou_seq.mean(axis=1)
    iou.extend(iou_seq.mean(axis=1).tolist())
iou = np.array(iou)
print "iou:", iou.mean()

with open("iou.txt", "w") as f:
    for fd, num in zip(subfd_names, iou):
        f.write("%s\t%f\n" % (fd, num))
    f.write("all\t%f\n" % iou.mean())

