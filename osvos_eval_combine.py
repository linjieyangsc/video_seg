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
gt_path = '/raid/ljyang/data/DAVIS/Annotations/480p_split'
pred_path = 'DAVIS/Results/Segmentations/480p/OSVOS_parent_flow'
pred_path_im = 'DAVIS/Results/Segmentations/480p/OSVOS'
sav_path = 'DAVIS/Visualize'
fds = os.listdir(pred_path)
fds = sorted(fds)
im_num = 0
iou =[] 
seq_n = 0
subfd_names = []
im_suffixes = ["_next_0","_next_1","_prev_0","_prev_1"]
for i, fd in enumerate(fds):
    print fd
    sub_fds = os.listdir(os.path.join(gt_path, fd))
    sub_fds = sorted(sub_fds)
    for sub_fd in sub_fds:
        subfd_names.append(fd + '/' + sub_fd)
        im_list = os.listdir(os.path.join(gt_path, fd, sub_fd))
        im_list = sorted(im_list)
        print sub_fd
        iou_seq = 0
        for im_name in im_list[1:-1]:
            label_gt = cv2.imread(os.path.join(gt_path, fd, sub_fd, im_name))
            label_gt = label_gt[:,:,0]
            label_gt = label_gt > 0
            # voting using 4 flow images
            score_pred_flow = np.zeros(label_gt.shape, dtype=np.float32)
            for suf in im_suffixes:
                #print im_name[:-4] + suf + '.png'
                #label_pred = cv2.imread(os.path.join(pred_path, fd, sub_fd, im_name[:-4] + suf + '.png'))
                score_pred = np.load(os.path.join(pred_path, fd, sub_fd, im_name[:-4] + suf + '.npy'))
                #label_pred = label_pred[:,:,0]
                #label_pred = label_pred > 0
                score_pred_flow += score_pred
            score_pred_flow /= 4
            score_pred_im = np.load(os.path.join(pred_path_im, fd, sub_fd, im_name[:-4] + '.npy'))
            score_pred_all = score_pred_im + score_pred_flow
            label_pred_all = score_pred_all > 1.0 # tunable threshold
            
            iou_seq += calcIoU(label_gt, label_pred_all)
        iou_seq /= len(im_list)
        iou.append(iou_seq)
iou = np.array(iou)
print "iou:", iou.mean()

print zip(subfd_names, iou)
with open("iou.txt", "w") as f:
    for fd, num in zip(subfd_names, iou):
        f.write("%s\t%f\n" % (fd, num))
    f.write("all\t%f\n" % iou.mean())

