import os
import sys
import cv2
import json
from PIL import Image
import numpy as np
from util import calcIoU
PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128]

data_path = sys.argv[1]
pred_path = sys.argv[2] #'DAVIS/Results/Segmentations/480p/OSVOS'
dataset_split = sys.argv[3]
if len(sys.argv) > 4:
    vis_path = sys.argv[4]
else:
    vis_path = None
listFile = '%s/%s_seqs.json' % (data_path, dataset_split)
gt_path = os.path.join(data_path, 'Annotations', '480p')
seq_data = json.load(open(listFile))
im_num = 0
iou =[]
seq_n = 0
sample_n = 0
subfd_names = []
for seq  in seq_data:
    vid_frames = seq['frames']
    vid_anno_path = seq['anno_path']
    vid_id = seq['vid']
    anno_path = os.path.join(data_path, vid_anno_path)
    # gather score and compute predicted label map
    for label_id, frames in vid_frames.iteritems():
        iou_seq = []
        im_list = frames[1:] # remove first and last image
        for i,im_name in enumerate(im_list):
            iou_im = 0
            scores = []
            label_gt = Image.open(os.path.join(anno_path, im_name))
            class_n = 1
            
            label_pred_im = Image.open(os.path.join(pred_path, vid_id, label_id, im_name))
            label_gt = label_gt.resize(label_pred_im.size)
            label_gt = (np.array(label_gt) == int(label_id))
            label_pred = np.array(label_pred_im) > 0
            #cv2.resize(label_pred, label_gt.shape, label_pred, 0, 0, cv2.INTER_NEAREST)
            if vis_path:
                res_im = Image.fromarray(label_pred, mode="P")
                res_im.putpalette(PALETTE)
                if not os.path.exists(os.path.join(vis_path, fd)):
                    os.makedirs(os.path.join(vis_path, fd))
                res_im.save(os.path.join(vis_path, fd, im_name))
            iou_seq.append(calcIoU(label_gt, label_pred, class_n))
        iou_seq = np.stack(iou_seq, axis=1)
        print iou_seq.mean(axis=1)
        iou.extend(iou_seq.mean(axis=1).tolist())#flatten and append
iou = np.array(iou)
print "iou:", iou.mean()
with open("iou.txt", "w") as f:
    for fd, num in zip(subfd_names, iou):
        f.write("%s\t%f\n" % (fd, num))
    f.write("all\t%f\n" % iou.mean())

