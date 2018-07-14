"""
Provides utility functions for OSMN library
"""
import os
import numpy as np
#from image_util import compute_robust_moments
from PIL import Image, ImageEnhance
import random
import cv2
PI = 3.1415926
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

def get_mask_bbox(m, border_pixels=0):
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

def perturb_mask(mask, center_perturb = 0.1, size_perturb=0.1):
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

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[:2])/2)
    rot_mat = cv2.getRotationMatrix2D(image_center,angle,1.0)
    angle_r = float(angle) / 180 * PI
    result = cv2.warpAffine(image, rot_mat, image.shape[:2],flags=cv2.INTER_NEAREST)
    return result

def get_scaled_box(box, out_size, in_size):
    box = np.array(box, dtype=np.float32)
    box[0::2] *= float(out_size[0])/in_size[0]
    box[1::2] *= float(out_size[1])/in_size[1]
    box = box.astype(np.int32)
    return box.tolist()

def adaptive_crop_box(mask, ext_ratio = 0.2):
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

def brightness_contrast_aug(im, brightness_range=(0.8, 1.3), contrast_range=(0.8, 1.3)):

    enhancer = ImageEnhance.Brightness(im)
    factor = np.random.uniform(brightness_range[0], brightness_range[1], 1)
    im = enhancer.enhance(factor)
    enhancer = ImageEnhance.Contrast(im)
    factor = np.random.uniform(contrast_range[0], contrast_range[1], 1)
    im = enhancer.enhance(factor)
    return im

def data_augmentation(im, label, new_size, 
        data_aug_flip = True, pad_ratio = 0, keep_aspect_ratio = False, 
        random_crop_ratio = 0, random_rotate_angle=0, color_aug=False):
    #old_size = im.size
    if random_crop_ratio > 0 or pad_ratio > 0:
        if random_crop_ratio > 0:
            crop_ratio = np.random.uniform( pad_ratio - random_crop_ratio, pad_ratio + random_crop_ratio, 4)
        elif pad_ratio > 0:
            crop_ratio = np.array([pad_ratio] * 4)
        crop_points = [0,0,im.size[0],im.size[1]]
        crop_points[0] = int(- crop_ratio[0] * im.size[0])
        crop_points[1] = int(- crop_ratio[1] * im.size[1])
        crop_points[2] += int(crop_ratio[2] * im.size[0])
        crop_points[3] += int(crop_ratio[3] * im.size[1])
        im = im.crop(crop_points)
        label = label.crop(crop_points)
    if keep_aspect_ratio:
        # resize but keeep aspect ratio
        ratio = np.amin(np.array(new_size, dtype=np.float32) / np.array(im.size))
        ka_size = (np.array(im.size) * ratio).astype(np.int32).tolist() 
        im = im.resize(ka_size, Image.BILINEAR)
        label = label.resize(ka_size, Image.NEAREST)
        padding_size = (np.array(new_size) - np.array(ka_size))/2
        padding_size_2 = np.array(new_size) - padding_size
        padding_pos = [ -padding_size[0], -padding_size[1], padding_size_2[0], padding_size_2[1]]
        im = im.crop(padding_pos)
        label = label.crop(padding_pos)
    else:
        im = im.resize(new_size, Image.BILINEAR)
        label = label.resize(new_size, Image.NEAREST)
    if color_aug:
        im = brightness_contrast_aug(im)
    if random_rotate_angle:
        angle = (random.random() - 0.5) * 2 * random_rotate_angle
        im = Image.fromarray(rotate_image(np.array(im), angle))
        label = Image.fromarray(rotate_image(np.array(label), angle))
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
