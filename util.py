"""
Provides utility functions for OSMN library
"""
import os
import numpy as np
#from image_util import compute_robust_moments
from PIL import Image
import random
import cv2
#import random
def get_dilate_structure(r):
    l = 2 * r + 1
    center = (r,r)
    x = np.arange(0,l)
    y = np.arange(0,l)
    nx, ny = np.meshgrid(x,y)
    coords = np.concatenate((nx[...,np.newaxis], ny[...,np.newaxis]), axis=2)
    s = np.sum((coords - center)**2, axis=2) <= r*r
    return s

def get_motion_blur_kernel(size):
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_type = random.choice((0,1,2,3))
    if kernel_type == 0:
        kernel_motion_blur[int((size-1)/2), :] = 1
    elif kernel_type == 1:
        kernel_motion_blur[:, int((size-1)/2)] = 1
    elif kernel_type == 2:
        kernel_motion_blur[range(size), range(size)] = 1
    else:
        kernel_motion_blur[range(size), range(size-1,-1,-1)] = 1
    kernel_motion_blur = kernel_motion_blur / size
    return kernel_motion_blur

def get_mask_bbox(m, border_pixels=8):
    if not np.any(m):
        # return a default bbox
        return (0, 0, m.shape[1], m.shape[0])
    rows = np.any(m, axis=1)
    cols = np.any(m, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    h,w = m.shape
    ymin = max(0, ymin - border_pixels)
    ymax = min(h-1, ymax + border_pixels)
    xmin = max(0, xmin - border_pixels)
    xmax = min(w-1, xmax + border_pixels)
    return (xmin, ymin, xmax, ymax)
def compute_robust_moments(binary_image, isotropic=False):
  index = np.nonzero(binary_image)
  points = np.asarray(index).astype(np.float32)
  if points.shape[1] == 0:
    return np.array([-1.0,-1.0],dtype=np.float32), \
        np.array([-1.0,-1.0],dtype=np.float32)
  points = np.transpose(points)
  points[:,[0,1]] = points[:,[1,0]]
  center = np.median(points, axis=0)
  if isotropic:
    diff = np.linalg.norm(points - center, axis=1)
    mad = np.median(diff)
    mad = np.array([mad,mad])
  else:
    diff = np.absolute(points - center)
    mad = np.median(diff, axis=0)
  std_dev = 1.4826*mad
  std_dev = np.maximum(std_dev, [5.0, 5.0])
  return center, std_dev
def get_gb_image(label, center_perturb = 0.2, std_perturb=0.4, blank_prob=0):
    if not np.any(label) or random.random() < blank_prob:
        #return a blank gb image
        center = np.array([label.shape[1]/2, label.shape[0]/2])
        std = np.array([label.shape[1]/2, label.shape[0]/2])
        return np.zeros((label.shape)), center, std
    center, std = compute_robust_moments(label)
    center_p_ratio = np.random.uniform(-center_perturb, center_perturb, 2)
    center_p = center_p_ratio * std + center
    std_p_ratio = np.random.uniform(1.0 / (1 + std_perturb), 1.0 + std_perturb, 2)
    std_p = std_p_ratio * std
    h,w = label.shape
    x = np.arange(0, w)
    y = np.arange(0, h)
    nx, ny = np.meshgrid(x,y)
    coords = np.concatenate((nx[...,np.newaxis], ny[...,np.newaxis]), axis = 2)
    normalizer = 0.5 /(std_p * std_p)
    D = np.sum((coords - center_p) ** 2 * normalizer, axis=2)
    D = np.exp(-D)
    D = np.clip(D, 0, 1)
    return D, center_p, std_p

def perturb_mask(mask, center_perturb = 0.1, size_perturb=0.05):
    if not np.any(mask):
        return np.zeros((mask.shape))
    xmin, ymin, xmax, ymax = get_mask_bbox(mask, border_pixels=0)
    mask_size = np.array((xmax - xmin, ymax - ymin))
    center = np.array(((xmin+xmax)/2, (ymin + ymax)/2))
    cropped_mask = mask[ymin:ymax+1,xmin:xmax+1]
    mask_out = np.zeros(mask.shape)
    out_size = np.array(mask_out.shape[1::-1],dtype=np.int32)
    size_ratio = np.random.uniform(1.0-size_perturb, 1.0 + size_perturb, 1)
    cropped_mask = cv2.resize(cropped_mask,(0,0),fx=size_ratio[0], fy=size_ratio[0], interpolation=cv2.INTER_NEAREST)
    size_p = np.array(cropped_mask.shape[1::-1], dtype=np.int32)
    size_p_1 = size_p / 2
    size_p_2 = size_p - size_p_1
    center_p_ratio = np.random.uniform(-center_perturb, center_perturb, 2)
    center_p = center_p_ratio * mask_size + center
    center_p = center_p.astype(np.int32)
    out_start = np.maximum(0, center_p - size_p_1)
    src_start = np.maximum(0, size_p_1 - center_p)
    out_end = np.minimum(out_size, center_p + size_p_2)
    src_end = np.minimum(size_p, size_p - (center_p + size_p_2 - out_size))
    mask_out[out_start[1]:out_end[1], out_start[0]:out_end[0]] = cropped_mask[src_start[1]:src_end[1], src_start[0]: src_end[0]]
    return mask_out

def adaptive_crop_box(mask, ext_ratio = 0.3):
    bbox = get_mask_bbox(mask, border_pixels=0)
    bbox_size = np.array([bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1])
    bbox_center = np.array([bbox[0]+bbox[2], bbox[1]+bbox[3]])/2
    p1 = np.array(bbox_center - bbox_size * (1+ext_ratio), dtype=np.int32)
    p1 = np.maximum(0, p1)
    p2 = np.array(bbox_center + bbox_size * (1+ext_ratio), dtype=np.int32)
    p2 = np.minimum(mask.shape[::-1], p2)
    return (p1[0], p1[1], p2[0], p2[1])

def to_bgr(image):
    if len(image.shape) < 3:
        image = np.repeat(image[...,np.newaxis], 3, axis=2)
    image = image[:,:, 2::-1]
    return image


def mask_image(image, label):
    assert(image.shape[:2] == label.shape)
    image[label == 0, :] = 0
    return image
def data_augmentation(im, label, new_size, 
        data_aug_flip = True, random_crop_ratio = 0):
    #old_size = im.size
    if random_crop_ratio:
        crop_pos = random.choice((0,1,2,3))
        crop_points = [0,0,im.size[0],im.size[1]]
        if crop_pos == 0:
            crop_points[0] = int(random_crop_ratio * im.size[0])
        elif crop_pos == 1:
            crop_points[1] = int(random_crop_ratio * im.size[1])
        elif crop_pos == 2:
            crop_points[2] -= int(random_crop_ratio * im.size[0])
        else:
            crop_points[3] -= int(random_crop_ratio * im.size[1])
        im = im.crop(crop_points)
        label = label.crop(crop_points)
    im = im.resize(new_size, Image.BILINEAR)
    label = label.resize(new_size, Image.NEAREST)
    #bbox = (bbox[0] * new_size[0] / float(old_size[0]),
    #        bbox[1] * new_size[1] / float(old_size[1]),
    #        bbox[2] * new_size[0] / float(old_size[0]),
    #        bbox[3] * new_size[0] / float(old_size[1]))
    if data_aug_flip:
        if random.random() > 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return im, label
def calcIoU(gt, pred, obj_n):
    assert(gt.shape == pred.shape)
    ious = np.zeros((obj_n), dtype=np.float32)
    for obj_id in range(1, obj_n+1):
        gt_mask = gt == obj_id
        pred_mask = pred == obj_id
        inter = gt_mask & pred_mask
    
        union = gt_mask | pred_mask
        if union.sum() == 0:
            ious[obj_id-1] = 1
        else:
            ious[obj_id-1] = float(inter.sum()) / union.sum()
    return ious
