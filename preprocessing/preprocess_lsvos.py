import sys
import os
from PIL import Image
import json
import numpy as np
data_dir = sys.argv[1]
#file_list = ['all_train.txt','all_val.txt','all_test.txt']
#file_list = ['test-seen.txt','test-unseen.txt']
file_list = ['val_subset.txt']
im_pixel_ratio_min = float(100)/256/448
seq_min_frames = 3
for fname in file_list:
    obj_frames_all = []
    with open(os.path.join(data_dir, fname)) as f:
        vid_list = [line.strip() for line in f]
    for vid_name in vid_list:
        
        vid_path = os.path.join(data_dir, vid_name, 'annotations')
        vid_im_list = sorted(os.listdir(vid_path), key=lambda x: int(x.split('.')[0]))
        vid_im_list= [x for x in vid_im_list if os.path.exists(os.path.join(
            data_dir, vid_name, 'images', x.split('.')[0]+'.jpg'))]
        obj_frames = {}
        obj_stats = {}
        for idx, imname in enumerate(vid_im_list):
            im = Image.open(os.path.join(vid_path, imname))
            im = np.array(im)
            max_id = im.max()
            for i in range(1,max_id+1):
                
                obj_pixels_ratio =float((im == i).sum())/im.size
                if i not in obj_stats:
                    obj_stats[i] = [idx, False]
                if obj_pixels_ratio > im_pixel_ratio_min and not obj_stats[i][1]:
                    obj_stats[i][1] = True
        for k,v in obj_stats.iteritems():
            if v[1] and v[0] < len(vid_im_list) - 1:
                obj_frames[k] = vid_im_list[v[0]:]
        if obj_frames: # not empty
            obj_frames_all.append({'vid':vid_name,'frames':obj_frames,'anno_path':vid_path,
                'image_path':os.path.join(data_dir, vid_name, 'images')})
    with open(os.path.join(data_dir, fname.split('.')[0]+'_seqs.json'),'w') as f:
        json.dump( obj_frames_all, f)

                    
            

