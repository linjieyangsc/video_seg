from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import cv2
import numpy as np

def save_result(input, output_path):
  PALETTE = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128,
             0, 128, 128, 128, 128, 128, 64, 0, 0, 191, 0, 0, 64, 128, 0, 191,
             128, 0, 64, 0, 128]
  im = Image.fromarray(input.astype(np.uint8), mode="P")
  im.putpalette(PALETTE)
  im.save(output_path)

def to_bgr(image):
  """Convert image to BGR format

    Args:
      image: Numpy array of uint8
    Returns:
      bgr: Numpy array of uint8
  """
  # gray scale image
  if image.ndim == 2:
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return bgr
  # BGRA format
  if image.shape[2] == 4:
    bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    return bgr
  bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  return bgr

def compute_opticalflow(prev_image, cur_image, args):
  prev_gray = cv2.cvtColor(prev_image, cv2.COLOR_RGB2GRAY)
  cur_gray  = cv2.cvtColor(cur_image, cv2.COLOR_RGB2GRAY)
  pyr_scale = args.pyr_scale
  pyr_levels = args.pyr_levels
  winsize = args.winsize
  iterations = args.iterations
  poly_n = args.poly_n
  poly_sigma = args.poly_sigma
  flow = cv2.calcOpticalFlowFarneback(prev_gray, cur_gray, flow=None,
                                      pyr_scale=pyr_scale,
                                      levels=pyr_levels,
                                      iterations=iterations,
                                      winsize=winsize,
                                      poly_n=poly_n,
                                      poly_sigma=poly_sigma,
                                      flags=0)
  return flow

def warp_flow(img, flow):
  h, w = flow.shape[:2]
  flow = -flow
  flow[:, :, 0] += np.arange(w)
  flow[:, :, 1] += np.arange(h)[:, np.newaxis]
  res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
  return res

def compute_moments(binary_image, isotropic=False):
  index = np.nonzero(binary_image)
  points = np.asarray(index).astype(np.float32)
  if points.shape[1] == 0:
    return np.array([-1.0,-1.0],dtype=np.float32), \
        np.array([-1.0, -1.0], dtype=np.float32)
  points = np.transpose(points)
  points[:,[0,1]] = points[:,[1,0]]
  center = np.mean(points, axis=0)
  diff = points - center
  diff = diff * diff
  dist = np.sqrt(np.sum(diff, axis=1))
  std_dev = np.std(dist)
  return center, std_dev

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

def compute_bbox(binary_image, isotropic=False):
  index = np.nonzero(binary_image)
  points = np.asarray(index).astype(np.float32)
  if points.shape[1] == 0:
    return np.array([-1.0,-1.0],dtype=np.float32), \
        np.array([-1.0,-1.0],dtype=np.float32)
  points = np.transpose(points)
  points[:,[0,1]] = points[:,[1,0]]
  sorted_x = np.sort(points[:, 0])
  sorted_y = np.sort(points[:, 1])
  dim_x = sorted_x[-1] - sorted_x[0]
  dim_y = sorted_y[-1] - sorted_y[0]
  center = np.array([(sorted_x[-1] + sorted_x[0])*0.5,
                     (sorted_y[-1] + sorted_y[0])*0.5])
  if isotropic:
    max_dim = np.maximum(dim_x,dim_y)
    dim = np.array([max_dim,max_dim])
  else:
    dim = np.array([dim_x,dim_y])
  return center, dim
