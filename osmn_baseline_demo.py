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
import osmn_baseline as osmn
from dataset_osmn import Dataset
os.chdir(root_folder)
baseDir = '/raid/ljyang/data'
# User defined parameters
train_path = os.path.join(baseDir, 'DAVIS/ImageSets/2017/train.txt')
val_path = os.path.join(baseDir, 'DAVIS/ImageSets/2017/val.txt')
with open(val_path, 'r') as f:
    val_seq_names = [line.strip() for line in f]
with open(train_path, 'r') as f:
    train_seq_names = [line.strip() for line in f]
test_imgs_with_guide = []
train_imgs_with_guide = []
baseDirImg = os.path.join(baseDir, 'DAVIS', 'JPEGImages', '480p')
baseDirLabel = os.path.join(baseDir, 'DAVIS', 'Annotations', '480p_split')
for name in val_seq_names:
    test_frames = sorted(os.listdir(os.path.join(baseDirImg, name)))
    label_fds = os.listdir(os.path.join(baseDirLabel, name))
    for label_id in label_fds:
        test_imgs_with_guide += [(os.path.join(baseDirImg, name, '00000.jpg'), 
                os.path.join(baseDirLabel, name, label_id, '00000.png'),
                os.path.join(baseDirImg, name, frame)) for frame in test_frames]
for name in train_seq_names:
    train_frames = sorted(os.listdir(os.path.join(baseDirImg, name)))
    label_fds = os.listdir(os.path.join(baseDirLabel, name))
    for label_id in label_fds:
        train_imgs_with_guide += [(os.path.join(baseDirImg, name, '00000.jpg'),
                os.path.join(baseDirLabel, name, label_id, '00000.png'),
                os.path.join(baseDirImg, name, frame),
                os.path.join(baseDirLabel, name, label_id, frame[:-4] + '.png')) 
                for frame in train_frames]
gpu_id = sys.argv[1]
train_model = True if len(sys.argv) > 2 else False
result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSMN_baseline')
# Train parameters
parent_path = os.path.join('models_src', 'OSVOS_parent', 'OSVOS_parent.ckpt-50000')
logs_path = 'models_osmn/baseline2'
max_training_iters = int(sys.argv[2])
# Define Dataset
dataset = Dataset(train_imgs_with_guide, test_imgs_with_guide, data_aug=True, data_aug_scales=[0.5, 0.8, 1])
# More training parameters
learning_rate = 0#1e-10
save_step = max_training_iters / 10
display_step = 10
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        osmn.train_finetune(dataset, parent_path, learning_rate, logs_path, max_training_iters,
                save_step, display_step, global_step, iter_mean_grad=1, ckpt_name='osmn')

# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        if train_model:
            checkpoint_path = os.path.join(logs_path, 'osmn.ckpt-'+str(max_training_iters))
        else:
            checkpoint_path = parent_path    
        osmn.test(dataset, checkpoint_path, result_path)
