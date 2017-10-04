"""
Provides utility functions for OSMN library
"""
import os
import numpy as np
#from image_util import compute_robust_moments
from PIL import Image
import random
#import random
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
def get_gb_image(label, center_perturb = 0.2, std_perturb=0.4, blank_prob=0.2):
    if not np.any(label) or random.random() < blank_prob:
        #return a blank gb image
        return np.zeros((label.shape))
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
    return D

def to_bgr(image):
    if len(image.shape) < 3:
        image = np.repeat(image[...,np.newaxis], 3, axis=2)
    image = image[:,:, 2::-1]
    return image


def mask_image(image, label):
    assert(image.shape[:2] == label.shape)
    image[label == 0, :] = 0
    return image
def data_augmentation(im, label, new_size, data_aug_flip):
    #old_size = im.size
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
            #bbox = (new_size[0] - bbox[1],
            #        new_size[0] - bbox[0],
            #       bbox[2], bbox[3])
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
