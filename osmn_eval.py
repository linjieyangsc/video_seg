"""
Test script
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
import osmn
from dataset_osmn import Dataset
os.chdir(root_folder)
baseDir = '/raid/ljyang/data'
# User defined parameters
val_path = os.path.join(baseDir, 'DAVIS/ImageSets/2017/train.txt')
with open(val_path, 'r') as f:
    val_seq_names = [line.strip() for line in f]
test_imgs_with_guide = []
baseDirImg = os.path.join(baseDir, 'DAVIS', 'JPEGImages', '480p')
baseDirLabel = os.path.join(baseDir, 'DAVIS', 'Annotations', '480p_split')
for name in val_seq_names:
    test_frames = sorted(os.listdir(os.path.join(baseDirImg, name)))
    label_fds = os.listdir(os.path.join(baseDirLabel, name))
    for label_id in label_fds:
        test_imgs_with_guide += [(os.path.join(baseDirImg, name, '00000.jpg'), 
                os.path.join(baseDirLabel, name, label_id, '00000.png'),
                os.path.join(baseDirImg, name, frame)) for frame in test_frames]
gpu_id = sys.argv[1]
result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSMN')
checkpoint_path = sys.argv[2]

# Define Dataset
dataset = Dataset([], test_imgs_with_guide)

# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        osmn.test(dataset, checkpoint_path, result_path)
