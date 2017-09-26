"""
Train / validation script
One-Shot Modulation Netowrk
"""
import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import matplotlib.pyplot as plt
import osmn_ss as osmn
from dataset_coco_ss import Dataset
baseDir = '/raid/ljyang/data'
# User defined parameters
val_path = os.path.join(baseDir, 'MS_COCO/val2017/{:012d}.jpg')
val_file = os.path.join(baseDir, 'MS_COCO/annotations/instances_val2017.json')
gpu_id = sys.argv[1]
result_path = os.path.join('COCO', 'OSMN')
checkpoint_path = sys.argv[2]
# Define Dataset
guide_image_mask = False
batch_size = 4
model_params = {'mod_last_conv':False}
dataset = Dataset(None, val_file, None, val_path, guide_image_mask=guide_image_mask, data_aug=False)

# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        osmn.test(dataset, model_params, checkpoint_path, result_path, batch_size = batch_size)
