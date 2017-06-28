import os
import numpy as np
import cv2
import sys
OPT_THRESH = 15
U_MAX = 0.3
V_MAX = 0.4
def normalize(flow):
    assert(flow.shape[2] == 2)
    flow[flow < -OPT_THRESH] = -OPT_THRESH
    flow[flow > OPT_THRESH] = OPT_THRESH
    flow = flow / OPT_THRESH
    return flow
def convert_yuv2bgr(yuv_image):
    assert(yuv_image.shape[2] == 3)
    yuv_image[:,:,1] *= U_MAX
    yuv_image[:,:,2] *= V_MAX
    bgr_image = np.zeros(yuv_image.shape, dtype=np.float32)
    bgr_image[:,:,0] = yuv_image[:,:,0] + 2.033 * yuv_image[:,:,1]
    bgr_image[:,:,1] = yuv_image[:,:,0] - 0.395 * yuv_image[:,:,1] - 0.581 * yuv_image[:,:,2]
    bgr_image[:,:,2] = yuv_image[:,:,0] + 1.14 * yuv_image[:,:,2]
    bgr_image = np.maximum(np.minimum(bgr_image * 255, 255), 0).astype(np.uint8)
    return bgr_image
imageDir = '/raid/ljyang/data/DAVIS/JPEGImages/480p'
saveVisDir = '/raid/ljyang/data/DAVIS/OpticalFlowVis/480p'
saveDir = '/raid/ljyang/data/DAVIS/OpticalFlow/480p'
seq_names = os.listdir(imageDir)
use_spatial_propagation = True
inst_dis = cv2.optflow.createOptFlow_DIS(cv2.optflow.DISOPTICAL_FLOW_PRESET_MEDIUM)
inst_dis.setUseSpatialPropagation(use_spatial_propagation)
inst_dis.setFinestScale(0)
inst_dis.setGradientDescentIterations(50)
inst_dis.setVariationalRefinementIterations(1)
inst_dis.setUseSpatialPropagation(True)
for seq in seq_names:
    print seq
    seqDir = os.path.join(imageDir, seq)
    saveFd = os.path.join(saveDir, seq)
    saveVisFd = os.path.join(saveVisDir, seq)
    
    if not os.path.exists(saveFd):
        os.makedirs(saveFd)
    if not os.path.exists(saveVisFd):
        os.makedirs(saveVisFd)
        
    image_list = os.listdir(seqDir)
    image_list = sorted(image_list)
    images = [cv2.imread(os.path.join(seqDir, image_name)) for image_name in image_list]
    gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    image_pairs = zip(image_list[:-1], gray_images[:-1], gray_images[1:])
    flow_im = np.ones((images[0].shape[0], images[0].shape[1], 3), dtype=np.float32) * 0.4
    print "generating next frame optical flow..."
    for image_name, image, image_next in image_pairs:
        #normalize to [-1,1]
        flow = inst_dis.calc(image, image_next, None)
        flow = normalize(flow)
        flow_im[:,:, 1:] = flow  # normalized to [-0.5,0.5]  
        flow_im_bgr = convert_yuv2bgr(flow_im)
        cv2.imwrite(os.path.join(saveVisFd, image_name[:-4] + '_next_0.png'), flow_im_bgr)
        flow_im[:,:, 1:] = -flow  # reverse
        flow_im_bgr_rev = convert_yuv2bgr(flow_im)
        cv2.imwrite(os.path.join(saveVisFd, image_name[:-4] + '_next_1.png'), flow_im_bgr_rev)
        np.save(os.path.join(saveFd,image_name[:-4] + '_next_0.npz'), flow)
        np.save(os.path.join(saveFd, image_name[:-4] + '_next_1.npz'), -flow)
    image_pairs = zip(image_list[1:], gray_images[1:], gray_images[:-1])
    print "generating prev frame optical flow..."
    for image_name, image, image_prev in image_pairs:
        flow = inst_dis.calc(image, image_prev, None)
        flow = normalize(flow)
        flow_im[:,:, 1:] = flow  # normalized to [-0.5,0.5]  
        flow_im_bgr = convert_yuv2bgr(flow_im)
        cv2.imwrite(os.path.join(saveVisFd, image_name[:-4] + '_prev_0.png'), flow_im_bgr)
        flow_im[:,:, 1:] = -flow # reverse
        flow_im_bgr_rev = convert_yuv2bgr(flow_im)
        cv2.imwrite(os.path.join(saveVisFd, image_name[:-4] + '_prev_1.png'), flow_im_bgr_rev)
        np.save(os.path.join(saveFd,image_name[:-4] + '_prev_0.npz'), flow)
        np.save(os.path.join(saveFd, image_name[:-4] + '_prev_1.npz'), -flow)

