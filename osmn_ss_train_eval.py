"""
Train / validation script
One-Shot Modulation Netowrk
Usage:
For training and validation:
    python osmn_train_eval.py [GPU_ID] [PARENT_MODEL_PATH] [TRAINING ITERS]
For validation only:
    python osmn_train_eval.py [GPU_ID] [MODEL_PATH]
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
import osmn_ss as osmn
from dataset_davis_ss import Dataset
os.chdir(root_folder)
baseDir = '/raid/ljyang/data'
# User defined parameters
train_path = os.path.join(baseDir, 'DAVIS/ImageSets/2017/train.txt')
val_path = os.path.join(baseDir, 'DAVIS/ImageSets/2017/val.txt')
with open(val_path, 'r') as f:
    val_seq_names = [line.strip() for line in f]
with open(train_path, 'r') as f:
    train_seq_names = [line.strip() for line in f]
use_static_guide = True # only use the first frame as appearance guide
test_imgs_with_guide = []
train_imgs_with_guide = []
baseDirImg = os.path.join(baseDir, 'DAVIS', 'JPEGImages', '480p')
baseDirLabel = os.path.join(baseDir, 'DAVIS', 'Annotations', '480p_split')
result_path = os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSMN')
for name in val_seq_names:
    test_frames = sorted(os.listdir(os.path.join(baseDirImg, name)))
    label_fds = os.listdir(os.path.join(baseDirLabel, name))
    for label_id in label_fds:
        test_imgs_with_guide += [(os.path.join(baseDirImg, name, '00000.jpg'), 
                os.path.join(baseDirLabel, name, label_id, '00000.png'),
                os.path.join(baseDirImg, name, '00001.jpg'))]
        if not use_static_guide:
            test_imgs_with_guide += [(os.path.join(baseDirImg, name, prev_frame), 
                os.path.join(result_path, name, label_id, prev_frame[:-4]+'.png'),
                    os.path.join(baseDirImg, name, frame)) 
                    for prev_frame, frame in zip(test_frames[1:-1], test_frames[2:])]
        else:
            test_imgs_with_guide += [(os.path.join(baseDirImg, name, '00000.jpg'),
                    os.path.join(baseDirLabel, name, label_id, '00000.png'),
                    os.path.join(result_path, name, label_id, prev_frame[:-4] +'.png'),
                    os.path.join(baseDirImg, name, frame))
                    for prev_frame, frame in zip(test_frames[1:-1], test_frames[2:])]
                    
for name in train_seq_names:
    train_frames = sorted(os.listdir(os.path.join(baseDirImg, name)))
    label_fds = os.listdir(os.path.join(baseDirLabel, name))
    for label_id in label_fds:
        if not use_static_guide:
            train_imgs_with_guide += [(os.path.join(baseDirImg, name, prev_frame),
                os.path.join(baseDirLabel, name, label_id, prev_frame[:-4] + '.png'),
                    os.path.join(baseDirImg, name, frame),
                    os.path.join(baseDirLabel, name, label_id, frame[:-4] + '.png')) 
                    for prev_frame, frame in zip(train_frames[:-1], train_frames[1:])]
        else:
            train_imgs_with_guide += [(os.path.join(baseDirImg, name, '00000.jpg'),
                os.path.join(baseDirLabel, name, label_id, '00000.png'),
                os.path.join(baseDirLabel, name, label_id, prev_frame[:-4] + '.png'),
                os.path.join(baseDirImg, name, frame),
                os.path.join(baseDirLabel, name, label_id, frame[:-4] + '.png'))
                for prev_frame, frame in zip(train_frames[:-1], train_frames[1:])]
gpu_id = sys.argv[1]
train_model = True if len(sys.argv) > 3 else False
# Train parameters
parent_path = sys.argv[2] #os.path.join('models_coco', 'osmn3', 'osmn.ckpt-45000')
logs_path = 'models_osmn/ss_static_guide'

# Define Dataset
dataset = Dataset(train_imgs_with_guide, test_imgs_with_guide, data_aug=True, data_aug_scales=[0.5, 0.8, 1])
# More training parameters
learning_rate = 1e-6
display_step = 10
batch_size = 4
model_params = {'mod_last_conv':False}
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        if train_model:
            max_training_iters = int(sys.argv[3])
            save_step = max_training_iters / 10
            global_step = tf.Variable(0, name='global_step', trainable=False)
            osmn.train_finetune(dataset, model_params, parent_path, None, learning_rate, logs_path, max_training_iters,
                             save_step, display_step, global_step, 
                             batch_size = batch_size,
                             iter_mean_grad=1, ckpt_name='osmn')

# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(gpu_id)):
        if train_model:
            checkpoint_path = os.path.join(logs_path, 'osmn.ckpt-'+str(max_training_iters))
        else:
            checkpoint_path = parent_path    
        osmn.test(dataset, model_params, checkpoint_path, result_path, batch_size=1)
