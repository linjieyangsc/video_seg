"""
Sergi Caelles (scaelles@vision.ee.ethz.ch)

This file is part of the OSVOS paper presented in:
    Sergi Caelles, Kevis-Kokitsi Maninis, Jordi Pont-Tuset, Laura Leal-Taixe, Daniel Cremers, Luc Van Gool
    One-Shot Video Object Segmentation
    CVPR 2017
Please consider citing the paper if you use this code.
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
import osvos_resnet
from dataset import Dataset
os.chdir(root_folder)
baseDir = '/raid/ljyang/data'
# User defined parameters
val_path = os.path.join(baseDir, 'DAVIS/ImageSets/2017/train.txt')
with open(val_path, 'r') as f:
    seq_names = [line.strip() for line in f]
for seq_name in seq_names:
    #label_fds = os.listdir(os.path.join(baseDir,'DAVIS/Annotations/480p_all', seq_name))
    gpu_id = sys.argv[1]
    result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS_train', seq_name)
    # Train parameters
    parent_path = os.path.join('models_flow', 'OSVOS_parent', 'OSVOS_parent.ckpt-10000')

    logs_path = os.path.join('models_flow', seq_name)

    # Define Dataset
    test_frames = sorted(os.listdir(os.path.join(baseDir, 'DAVIS', 'JPEGImages', '480p', seq_name)))
    test_imgs = [os.path.join(baseDir, 'DAVIS', 'JPEGImages', '480p', seq_name, frame) for frame in test_frames]
    dataset = Dataset(None, test_imgs, './')

    # Test the network
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            checkpoint_path = parent_path    
            osvos_resnet.test(dataset, checkpoint_path, result_path)
