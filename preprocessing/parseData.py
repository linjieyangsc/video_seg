import os
import sys
import cv2
from sets import Set
import numpy as np
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
        im = cv2.imread(im_path)
        h,w, ch = im.shape
        im = im.astype(np.int32)
        im_hash = im[:,:,0] * 256 * 256 + im[:,:,1] * 256 + im[:,:,2]
        all_hash.append(im_hash)
        color_set = Set(np.unique(im_hash))
        colors |= color_set
    colors.remove(0)
    print '%s has %d objects' % (fd, len(colors))
    print 'color hashes:'
    print colors
    for i, color in enumerate(list(colors)):
        split_dir = save_dir + fd + '/%d' % (i+1)
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        for im_name, hash_map in zip(im_list, all_hash):
            binary_map = hash_map == color
            mask_image = (binary_map * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(split_dir, im_name), mask_image)

        

