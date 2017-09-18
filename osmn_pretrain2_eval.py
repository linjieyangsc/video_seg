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
# Import OSVOS files
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
import osmn_pretrain2 as osmn
from dataset_coco import Dataset
os.chdir(root_folder)
baseDir = '/raid/ljyang/data'
# User defined parameters
val_path = os.path.join(baseDir, 'MS_COCO/val2017/{:012d}.jpg')
val_file = os.path.join(baseDir, 'MS_COCO/annotations/instances_val2017.json')
gpu_id = sys.argv[1]
result_path = os.path.join('COCO', 'OSMN2')
# Train parameters
checkpoint_path = sys.argv[2]
# Define Dataset
dataset = Dataset(None, val_file, None, val_path, data_aug=False)
# More training parameters
batch_size = 10

# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        osmn.test(dataset, checkpoint_path, result_path)
