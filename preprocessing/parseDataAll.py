import os
import sys
import cv2
from sets import Set
import numpy as np
anno_dir = 'Annotations/480p/'
save_dir = 'Annotations/480p_all/'
fds = os.listdir(anno_dir)
for fd in fds:
    im_list = os.listdir(anno_dir + fd)
    im_list = [item for item in im_list if item[-3:] == 'png']
    colors = Set()
    all_hash = []
    sub_dir = save_dir + fd
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
    for item in im_list:
        im_path = os.path.join(anno_dir + fd, item)
        im = cv2.imread(im_path)
        h,w, ch = im.shape
        im = im.astype(np.int32)
        binary_map = im.sum(axis=2) > 0
        
        mask_image = (binary_map * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(sub_dir, item), mask_image)

        

