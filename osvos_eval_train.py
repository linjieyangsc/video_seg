import os
import sys
import cv2
import numpy as np
def calcIoU(gt, pred):
    assert(gt.shape == pred.shape)
    inter = gt & pred
    
    union = gt | pred
    iou = float(inter.sum()) / union.sum()
    return iou
gt_path = '/raid/ljyang/data/DAVIS/Annotations/480p_all'
pred_path = 'DAVIS/Results/Segmentations/480p/OSVOS_train'
sav_path = 'DAVIS/Visualize'
fds = os.listdir(pred_path)
fds = sorted(fds)
im_num = 0
iou =[] 
seq_n = 0
for i, fd in enumerate(fds):
    print fd
    im_list = os.listdir(os.path.join(gt_path, fd))
    im_list = sorted(im_list)
    iou_seq = 0
    for im_name in im_list[1:-1]:
        label_gt = cv2.imread(os.path.join(gt_path, fd, im_name))
        label_pred = cv2.imread(os.path.join(pred_path, fd, im_name))
        label_gt = label_gt[:,:,0]
        label_gt = label_gt > 0
        label_pred = label_pred[:,:,0]
        label_pred = label_pred > 0
        
        iou_seq += calcIoU(label_gt, label_pred)
    iou_seq /= len(im_list)
    iou.append(iou_seq)
iou = np.array(iou)
print "iou:", iou.mean()

print zip(fds, iou)
with open("iou_train.txt", "w") as f:
    for fd, num in zip(fds, iou):
        f.write("%s\t%f\n" % (fd, num))
    f.write("all\t%f\n" % iou.mean())

