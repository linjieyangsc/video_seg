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
import osvos
from dataset import Dataset
os.chdir(root_folder)
baseDir = '/raid/ljyang/data'
# User defined parameters
val_path = os.path.join(baseDir, 'DAVIS/ImageSets/2017/val.txt')
with open(val_path, 'r') as f:
    seq_names = [line.strip() for line in f]
for seq_name in seq_names:
    gpu_id = sys.argv[1]
    train_model = True if len(sys.argv) > 2 else False
    label_fds = os.listdir(os.path.join(baseDir,'DAVIS/Annotations/480p_split', seq_name))
    for label_id in label_fds:
        if train_model:
            result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS_flow', seq_name, label_id)
        else:
            result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS_parent_flow', seq_name, label_id)
        # Train parameters
        parent_path = os.path.join('models_flow', 'OSVOS_parent_vgg', 'OSVOS_parent.ckpt-20000')
        logs_path = os.path.join('models_flow', seq_name, label_id)
        if train_model:
            max_training_iters = int(sys.argv[2])

        # Define Dataset
        test_frames = sorted(os.listdir(os.path.join(baseDir, 'DAVIS', 'OpticalFlowVis', '480p', seq_name)))
        test_imgs = [os.path.join(baseDir, 'DAVIS', 'OpticalFlowVis', '480p', seq_name, frame) for frame in test_frames]
        if train_model:
            train_imgs = []
            for i in range(2):

                train_imgs.append(os.path.join(baseDir, 'DAVIS', 'OpticalFlowVis', '480p', seq_name, '00000_next_%d.png' % i)+' '+
                          os.path.join(baseDir, 'DAVIS', 'Annotations', '480p_split', seq_name, label_id, '00000.png'))
            print train_imgs
            print label_id
            dataset = Dataset(train_imgs, test_imgs, './', data_aug=True, data_aug_scales=[0.5, 0.8, 1])
        else:
            dataset = Dataset(None, test_imgs, './')
        # Train the network
        if train_model:
            # More training parameters
            learning_rate = 1e-7
            side_supervision = 3
            save_step = max_training_iters
            display_step = 10
            with tf.Graph().as_default():
                with tf.device('/gpu:' + str(gpu_id)):
                    global_step = tf.Variable(0, name='global_step', trainable=False)
                    osvos.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters,
                                         save_step, display_step, global_step, iter_mean_grad=1, ckpt_name=seq_name)

        # Test the network
        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(gpu_id)):
                if train_model:
                    checkpoint_path = os.path.join('models_flow', seq_name, label_id, seq_name+'.ckpt-'+str(max_training_iters))
                else:
                    checkpoint_path = parent_path    
                osvos.test(dataset, checkpoint_path, result_path)
