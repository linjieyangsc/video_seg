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
import tensorflow as tf
slim = tf.contrib.slim
# Import OSVOS files
root_folder = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(root_folder))
import osvos
from dataset import Dataset

# User defined parameters
if len(sys.argv) < 2:
    print "Usage: python osvos_parent_demo_resnet.py [GPU_ID]"
    exit()
gpu_id = sys.argv[1]

# Training parameters
imagenet_ckpt = 'models/vgg_16.ckpt'
#imagenet_ckpt = 'models_resnet/OSVOS_parent/OSVOS_parent.ckpt-5000'
logs_path = os.path.join(root_folder, 'models_flow', 'OSVOS_parent_vgg')
store_memory = False
data_aug = False
iter_mean_grad = 10
max_training_iters = 20000
save_step = 5000
test_image = None
display_step = 100
ini_learning_rate = 1e-8
boundaries = [10000, 20000]
values = [ini_learning_rate, ini_learning_rate * 0.1]

# Define Dataset
train_file = 'train_parent_list_flow.txt'
val_file = 'val_list_flow.txt'
dataset_path = '/raid/ljyang/data/DAVIS'
dataset = Dataset(train_file, val_file, dataset_path, store_memory=store_memory, data_aug=data_aug)
# 
# Train the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
        osvos.train_parent(dataset, imagenet_ckpt, 1, learning_rate, logs_path, max_training_iters, save_step,
                           display_step, global_step, iter_mean_grad=iter_mean_grad, test_image_path=test_image,
                           ckpt_name='OSVOS_parent')

