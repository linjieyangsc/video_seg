import os
import sys
from PIL import Image
from sets import Set
import numpy as np
import cv2
anno_dir = 'Annotations/480p/'
save_dir = 'Annotations/480p_split/'
fds = os.listdir(anno_dir)
for fd in fds:
    im_list = os.listdir(anno_dir + fd)
    im_list = [item for item in im_list if item[-3:] == 'png']
    colors = Set()
    all_hash = []
    for item in im_list:
        im_path = os.path.join(anno_dir + fd, item)
        im = np.array(Image.open(im_path))
        cls_n = im.max()
        for i in range(1, cls_n+1):
            split_dir = save_dir + fd + '/%d' % (i)
            if not os.path.exists(split_dir):
                os.makedirs(split_dir)
            binary_map = im == i
            mask_image = (binary_map * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(split_dir, item), mask_image)

        

