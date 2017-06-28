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
import osvos_multi as osvos
from dataset import Dataset
os.chdir(root_folder)
baseDir = '/raid/ljyang/data'
# User defined parameters
val_path = os.path.join(baseDir, 'DAVIS/ImageSets/2017/val.txt')
with open(val_path, 'r') as f:
    seq_names = [line.strip() for line in f]
for seq_name in seq_names:
    print seq_name
    gpu_id = sys.argv[1]
    result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSVOS_multi_progressive', seq_name)
    # Train parameters
    parent_path = os.path.join('models_src', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
    logs_path = os.path.join('models_prog', seq_name)

    
    # Define Dataset
    all_frames = sorted(os.listdir(os.path.join(baseDir, 'DAVIS', 'JPEGImages', '480p', seq_name)))
    all_imgs = [os.path.join(baseDir, 'DAVIS', 'JPEGImages', '480p', seq_name, frame) for frame in all_frames]
    # Get class number
    train_im = Image.open(os.path.join(baseDir, 'DAVIS', 'Annotations', '480p', seq_name, '00000.png'))
    cls_n = np.array(train_im).max() + 1
    print 'seq %s has %d classes' % (seq_name, cls_n) 
    for stage, train_im in enumerate(all_frames[:-1]): 
        if stage == 0:
            train_label = os.path.join(baseDir, 'DAVIS', 'Annotations', '480p', seq_name, '00000.png')
        else:
            train_label = os.path.join(result_path, all_frames[stage][:-4] + '.png')
        train_imgs = [all_imgs[stage]+' '+train_label]
        test_imgs = [all_imgs[stage+1]]
        dataset = Dataset(train_imgs, test_imgs, './', data_aug=True, data_aug_scales=[0.8, 1, 1.2])
        if stage==0:
            max_training_iters = int(sys.argv[2])
        else:
            parent_path = os.path.join('models_prog', seq_name, seq_name + '-' + str(stage) + '.ckpt-' + str(max_training_iters))
            max_training_iters = int(sys.argv[3])
        # Train the network
        checkpoint_path = os.path.join('models_prog', seq_name, seq_name + '-' + str(stage+1) + '.ckpt-' + str(max_training_iters) )
   
        if not os.path.exists(checkpoint_path+ '.meta'):
            # More training parameters
            learning_rate = 1e-10
            save_step = max_training_iters
            side_supervision = 3
            display_step = 10
            with tf.Graph().as_default():
                with tf.device('/gpu:' + str(gpu_id)):
                    global_step = tf.Variable(0, name='global_step', trainable=False)
                    osvos.train_finetune(dataset, parent_path, side_supervision, learning_rate, logs_path, max_training_iters,
                                         save_step, display_step, global_step, iter_mean_grad=1, progressive=(stage>0), 
                                         ckpt_name=seq_name+'-' + str(stage+1), n_outputs=cls_n)

        # Test the network
        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(gpu_id)):
                osvos.test(dataset, checkpoint_path, result_path, use_gf=True, n_outputs=cls_n)
