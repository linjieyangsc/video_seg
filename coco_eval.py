import os
import sys
import cv2
from PIL import Image
import cPickle
import numpy as np
from util import calcIoU
sys.path.append('../coco/PythonAPI')
from pycocotools.coco import COCO
test_anno_file = '/raid/ljyang/data/MS_COCO/annotations/instances_val2017.json'
gt_annos = cPickle.load(open('cache/val_annos.pkl', 'rb'))
print 'obj number', len(gt_annos)
test_data = COCO(test_anno_file)
pred_path = sys.argv[1] #'DAVIS/Results/Segmentations/480p/OSVOS'
pred_im_num =len([name for name in os.listdir(pred_path) if len(name) == 10 and name[-4:]=='.png'])
im_num = min(len(gt_annos), pred_im_num)
print 'Image number:', im_num
iou =[] 
seq_n = 0
subfd_names = []
new_size = (400,400)
for i in range(im_num):
    im_name = '%06d.png' % i
    anno = gt_annos[i]
    label_gt = test_data.annToMask(anno).astype(np.uint8)
    label_gt = np.array(Image.fromarray(label_gt).resize(new_size))
    label_pred = Image.open(os.path.join(pred_path, im_name))
    label_pred = np.array(label_pred, dtype=np.uint8) > 0
    iou.append(calcIoU(label_gt, label_pred, 1))
iou = np.array(iou)
print 'iou shape', iou.shape
print "iou:", iou.mean()


