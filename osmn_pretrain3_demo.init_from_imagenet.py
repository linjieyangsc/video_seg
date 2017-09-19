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
import osmn_pretrain3_init as osmn
from dataset_coco import Dataset
os.chdir(root_folder)
baseDir = '/raid/ljyang/data'
# User defined parameters
train_path = os.path.join(baseDir, 'MS_COCO/train2017/{:012d}.jpg')
val_path = os.path.join(baseDir, 'MS_COCO/val2017/{:012d}.jpg')
train_file = os.path.join(baseDir, 'MS_COCO/annotations/instances_train2017.json')
val_file = os.path.join(baseDir, 'MS_COCO/annotations/instances_val2017.json')
gpu_id = sys.argv[1]
result_path = os.path.join('COCO', 'OSMN3')
# Train parameters
init_vgg_path = os.path.join('models', 'vgg_16.ckpt')
parent_path = init_vgg_path #os.path.join('models_src', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
logs_path = 'models_coco/osmn3_init'
max_training_iters = int(sys.argv[2])

# Define Dataset
dataset = Dataset(train_file, val_file, train_path, val_path, data_aug=True, data_aug_scales=[0.8, 1, 1.2])
# More training parameters
learning_rate = 1e-3
save_step = max_training_iters / 20
display_step = 10
batch_size = 8
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        osmn.train_finetune(dataset, init_vgg_path, parent_path, learning_rate, logs_path, max_training_iters,
                             save_step, display_step, global_step, batch_size = batch_size, 
                             iter_mean_grad=1, ckpt_name='osmn')

# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        checkpoint_path = os.path.join(logs_path, 'osmn.ckpt-'+str(max_training_iters))
        osmn.test(dataset, checkpoint_path, result_path, batch_size = batch_size)
