"""
Script to merge multi-object results for YoutubeVOS
""" 
import os
import sys
import cv2
import json
from PIL import Image
import numpy as np
from util import calcIoU
# Any PALLETTE works
PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191, 128, 0, 64, 0, 128]
data_path = sys.argv[1]
pred_path = sys.argv[2] 
dataset_split = sys.argv[4]
merge_path = sys.argv[3]
listFile = '%s/%s/meta.json' % (data_path, dataset_split)
seq_data = json.load(open(listFile))['videos']
im_num = 0
iou =[]
seq_n = 0
sample_n = 0
prediction_size = (448, 256)
subfd_names = []
for vid_id, seq  in  seq_data.iteritems():
    print 'processing', vid_id
    vid_frames = seq['objects']
    vid_anno_path = os.path.join(dataset_split, 'Annotations', vid_id)
    anno_path = os.path.join(data_path, vid_anno_path)
    anno_files = os.listdir(anno_path)
    sample_anno = Image.open(os.path.join(anno_path, anno_files[0]))
    width,height = sample_anno.size
    save_path = os.path.join(merge_path, vid_anno_path)
    # gather score and compute predicted label map
    frame_to_obj_dict = {}
    for label_id, obj_info in vid_frames.iteritems():
        frames = obj_info['frames']
        for im_name in frames:
            if im_name in frame_to_obj_dict:
                frame_to_obj_dict[im_name].append(int(label_id))
            else:
                frame_to_obj_dict[im_name] = [int(label_id)]
    for im_name, obj_ids in frame_to_obj_dict.iteritems():
        scores = []
        for label_id in obj_ids:
            score_path = os.path.join(pred_path, vid_id, str(label_id), im_name+'.npy')
            if not os.path.exists(score_path):
                # no predicted score file, which means it is first frame for the 
                # corresponding object, read first frame gt label
                gt = Image.open(os.path.join(anno_path, im_name+'.png'))
                gt = gt.resize(prediction_size, Image.NEAREST)
                gt = np.array(gt)
                score = (gt == label_id).astype(np.float32)
            else:
                score = np.load(open(os.path.join(pred_path, vid_id, str(label_id), im_name+'.npy')))
                
            print score.shape
            scores.append(score)
        obj_ids_ext = np.array([0] + obj_ids, dtype=np.uint8)
        im_size = scores[0].shape
        bg_score = np.ones(im_size) * 0.5
        scores = [bg_score] + scores
        score_all = np.stack(tuple(scores), axis = -1)
        class_n = score_all.shape[2] - 1
        pred_idx = score_all.argmax(axis=2)
        label_pred = obj_ids_ext[pred_idx]
        
        res_im = Image.fromarray(label_pred, mode="P")
        res_im.putpalette(PALETTE)
        res_im = res_im.resize((width,height),Image.NEAREST)
        if not os.path.exists(os.path.join(save_path)):
            os.makedirs(os.path.join(save_path))
        res_im.save(os.path.join(save_path, im_name+'.png'))
