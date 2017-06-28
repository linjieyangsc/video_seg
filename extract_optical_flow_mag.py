import os
import numpy as np
import cv2
import sys
OPT_THRESH = 40
def normalize_mag(flow):
    assert(flow.shape[2] == 2)
    flow[:,:,0] -= flow[:,:,0].mean()
    flow[:,:,1] -= flow[:,:,1].mean()
    flow = np.sqrt(flow[:,:,0] ** 2 + flow[:,:,1] ** 2)
    flow[flow > OPT_THRESH] = OPT_THRESH
    flow = flow / OPT_THRESH
    return flow
imageDir = '/raid/ljyang/data/DAVIS/JPEGImages/Full-Resolution'
imageReDir ='/raid/ljyang/data/DAVIS/JPEGImages/480p'
saveVisDir = '/raid/ljyang/data/DAVIS/OpticalFlowMag/480p'
seq_names = os.listdir(imageDir)
use_spatial_propagation = True
inst_dis = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_MEDIUM)
inst_dis.setUseSpatialPropagation(use_spatial_propagation)
inst_dis.setFinestScale(0)
inst_dis.setGradientDescentIterations(100)
inst_dis.setVariationalRefinementIterations(1)
inst_dis.setUseSpatialPropagation(True)
for seq in seq_names:
    seqDir = os.path.join(imageDir, seq)
    saveVisFd = os.path.join(saveVisDir, seq)
    
    if not os.path.exists(saveVisFd):
        os.makedirs(saveVisFd)
        
    image_list = os.listdir(seqDir)
    image_list = sorted(image_list)
    images= []
    #test_im = cv2.imread(os.path.join(imageReDir, seq, '00000.jpg'))
    #re_h, re_w, ch = test_im.shape
    
    #if re_w == 854:
        # previously correct
    #    continue
    
    print 'generate flow for', seq
    #print 'im size', re_h, re_w
    for image_name in image_list:
        im = cv2.imread(os.path.join(seqDir, image_name))
        images.append(cv2.resize(im, (re_w * 2, re_h * 2)))
    gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    image_pairs = zip(image_list[:-1], gray_images[:-1], gray_images[1:])
    print "generating next frame optical flow..."
    for image_name, image, image_next in image_pairs:
        flow = inst_dis.calc(image, image_next, None)
        flow = normalize_mag(flow)
        flow = cv2.resize(flow, (re_w, re_h))
        cv2.imwrite(os.path.join(saveVisFd, image_name[:-4] + '.png'), (flow * 255).astype(np.uint8))

