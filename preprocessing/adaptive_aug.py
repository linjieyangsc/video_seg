import os
import numpy as np
from PIL import Image
MAX_RATIO = 3
MIN_RATIO = 0.5
#RESIZE_RATIO = 4.5 # from 1080 to 480
TARGET_H = 480
TARGET_W = 854
def bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return [cmin, rmin, cmax, rmax]

setFile = 'ImageSets/2017/val.txt'
imDir = 'JPEGImages/Full-Resolution/'
labelDir = 'Annotations/Full-Resolution/'
imSavDir = 'JPEGImages/480p_train_aug/'
labelSavDir = 'Annotations/480p_train_aug/'
size_multi = 4
with open(setFile, 'r') as f:
	seq_names = [line.strip() for line in f]
for seq_name in seq_names:
	label = Image.open(os.path.join(labelDir, seq_name, '00000.png'))
	label_data = np.array(label)
	im = Image.open(os.path.join(imDir, seq_name, '00000.jpg'))

	im_data = np.array(im)
	h, w, ch = im_data.shape
	obj_n = label_data.max()
	large_ratio_all = MIN_RATIO
	min_ratio_all = MAX_RATIO
	RESIZE_RATIO = float(h) / TARGET_H
	if not os.path.exists(os.path.join(imSavDir, seq_name)):
		os.makedirs(os.path.join(imSavDir, seq_name))
	if not os.path.exists(os.path.join(labelSavDir, seq_name)):
		os.makedirs(os.path.join(labelSavDir, seq_name))
	for obj_id in range(1, obj_n+1):
		obj_mask = label_data == obj_id
		bbox = np.array(bbox2(obj_mask))
		bbox_w = (bbox[2] - bbox[0])
		bbox_h = bbox[3] - bbox[1]
		large_ratio = min(float(w) / bbox_w, float(h)/bbox_h, MAX_RATIO)
		min_ratio = max(min(float(w) / 4 / bbox_w, float(h) / 4 / bbox_h), MIN_RATIO)
		# ratio <=1 will be conducted on the whole image
		large_ratio_all = max(large_ratio, large_ratio_all)
		min_ratio_all = min(min_ratio_all, min_ratio)
		if large_ratio > 1.3:
			# one or two scales
			if large_ratio > 2:
				scales = [np.sqrt(large_ratio), large_ratio]
			else:
				scales = [large_ratio]
			for scale in scales:
				adj_scale = scale/RESIZE_RATIO
				adj_w = int(w * adj_scale/ size_multi) * size_multi
				adj_h = int(h * adj_scale/size_multi) * size_multi
				print 'scale', scale
				print 'image size',adj_w, adj_h
				im_re = im.resize((adj_w, adj_h), Image.BILINEAR)
				label_re = label.resize((adj_w, adj_h), Image.NEAREST)
				adj_bbox = bbox * adj_scale
				crop_x1 = int(min(max(0, (adj_bbox[0] + adj_bbox[2])/2 - TARGET_W / 2),
					adj_w - TARGET_W))
				crop_x2 = crop_x1 + TARGET_W 
				crop_y1 = int(min(max(0, (adj_bbox[1] + adj_bbox[3])/2 - TARGET_H / 2),
					adj_h - TARGET_H))
				crop_y2 = crop_y1 + TARGET_H 

				print (crop_x1, crop_y1, crop_x2, crop_y2)
				im_crop = im_re.crop((crop_x1, crop_y1, crop_x2, crop_y2))
				im_name = '00000_%d_%.02f' % (obj_id, scale)
				im_crop.save(os.path.join(imSavDir, seq_name, im_name + '.jpg'))
				label_crop = label_re.crop((crop_x1, crop_y1, crop_x2, crop_y2))
				label_crop.save(os.path.join(labelSavDir, seq_name, im_name + '.png'))
	if min_ratio_all < 0.8:
		if min_ratio_all < 0.7:
			scales = [min_ratio_all, np.sqrt(min_ratio_all)]
		else:
			scales = [min_ratio_all]
		for scale in scales:
			adj_scale = scale/RESIZE_RATIO
			adj_w = int(w * adj_scale/size_multi) * size_multi
			adj_h = int(h * adj_scale/size_multi) * size_multi
			print 'scale', scale
			print 'image size',adj_w, adj_h
			im_re = im.resize((adj_w, adj_h), Image.BILINEAR)
			label_re = label.resize((adj_w, adj_h), Image.NEAREST)
			im_name = '00000_%.02f' % (scale)
			im_re.save(os.path.join(imSavDir, seq_name, im_name + '.jpg'))
			label_re.save(os.path.join(labelSavDir, seq_name, im_name + '.png'))
	adj_scale = 1.0/RESIZE_RATIO
	adj_w = int(w * adj_scale/ size_multi) * size_multi
	adj_h = int(h * adj_scale/size_multi) * size_multi
	im_re = im.resize((adj_w, adj_h), Image.BILINEAR)
	label_re = label.resize((adj_w, adj_h), Image.NEAREST)
	im_re.save(os.path.join(imSavDir, seq_name, '00000.jpg'))
	label_re.save(os.path.join(labelSavDir, seq_name, '00000.png'))





