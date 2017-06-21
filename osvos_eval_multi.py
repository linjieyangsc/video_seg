import os
import sys
from PIL import Image
import numpy as np
def calcIoU(gt, pred):
    assert(gt.shape == pred.shape)
    obj_n = max(gt.max(), pred.max())
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
    return ious.mean()

gt_path = '/raid/ljyang/data/DAVIS/Annotations/480p'
pred_path = 'DAVIS/Results/Segmentations/480p/OSVOS'
sav_path = 'DAVIS/Visualize'
listFile = '/raid/ljyang/data/DAVIS/ImageSets/2017/val.txt'
with open(listFile, 'r') as f:
    fds = [line.strip() for line in f]
im_num = 0
iou =[] 
seq_n = 0
for i, fd in enumerate(fds):
    print fd
    file_list = os.listdir(os.path.join(gt_path, fd))
    im_list = [f for f in file_list if len(f) > 4 and f[-4:] == '.png']
    im_list = sorted(im_list)
    iou_seq = 0
    for im_name in im_list[1:-1]:
        label_gt = np.array(Image.open(os.path.join(gt_path, fd, im_name)))
        label_pred = np.array(Image.open(os.path.join(pred_path, fd, im_name)))
        if (label_pred.shape == 3):
            print "pred label should only have 1 channel"
            exit()
        iou_seq += calcIoU(label_gt, label_pred)
    iou_seq /= len(im_list)-2
    print 'iou:', iou_seq
    iou.append(iou_seq)
iou = np.array(iou)
print "iou:", iou.mean()

with open("iou.txt", "w") as f:
    for fd, num in zip(fds, iou):
        f.write("%s\t%f\n" % (fd, num))
    f.write("all\t%f\n" % iou.mean())

