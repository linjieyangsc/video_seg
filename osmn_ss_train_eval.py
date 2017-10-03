"""
Train / validation script
One-Shot Modulation Netowrk
Usage:
For training and validation:
    python osmn_train_eval.py [GPU_ID] [PARENT_MODEL_PATH] [RESULT_PATH] [MODEL_SAVE_PATH] [TRAINING ITERS]
For validation only: 
    python osmn_train_eval.py [GPU_ID] [MODEL_PATH] [RESULT_PATH]
"""
import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import argparse
import osmn_ss as osmn
from dataset_davis_ss import Dataset

def add_arguments(parser):
    group = parser.add_argument_group(title='Paths Arguments')
    group.add_argument(
            '--data_path',
            type=str,
            required=False,
            default='/raid/ljyang/data/DAVIS')
    group.add_argument(
            '--src_model_path',
            type=str,
            required=True,
            default='')
    group.add_argument(
            '--result_path',
            type=str,
            required=True,
            default='')
    group.add_argument(
            '--model_save_path',
            type=str,
            required=False,
            default='')


    group = parser.add_argument_group(title='Model Arguments')
    group.add_argument(
            '--use_static_guide',
            required=False,
            action='store_false',
            default=True,
            help="""
                only use the first frame as visual guide or use the previous frame as visual guide
                """)
    group.add_argument(
            '--mod_last_conv',
            required=False,
            action='store_true',
            default=False)
    group.add_argument(
            '--sp_late_fusion',
            required=False,
            action='store_true',
            default=False)
    group = parser.add_argument_group(title='Data Argument')
    group.add_argument(
            '--data_aug_scales',
            type=list,
            required=False,
            default=[0.5,0.8,1])
    group.add_argument(
            '--batch_size',
            type=int,
            required=False,
            default=4)

    group = parser.add_argument_group(title='Running Arguments')
    group.add_argument(
            '--gpu_id',
            type=int,
            required=False,
            default=0)
    group.add_argument(
            '--training_iters',
            type=int,
            required=False,
            default=10000)
    group.add_argument(
            '--save_iters',
            type=int,
            required=False,
            default=1000)
    group.add_argument(
            '--learning_rate',
            type=float,
            required=False,
            default=1e-6)
    group.add_argument(
            '--display_iters',
            type=int,
            required=False,
            default=10)
    group.add_argument(
            '--is_training',
            required=False,
            action='store_false',
            default=True,
            help="""\
                is it training or testing?
                """)
    group.add_argument(
            '--resume_training',
            required=False,
            action='store_false',
            default=True)

parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()
print args
baseDir = args.data_path
# User defined parameters
train_path = os.path.join(baseDir, 'ImageSets/2017/train.txt')
val_path = os.path.join(baseDir, 'ImageSets/2017/val.txt')
with open(val_path, 'r') as f:
    val_seq_names = [line.strip() for line in f]
with open(train_path, 'r') as f:
    train_seq_names = [line.strip() for line in f]
use_static_guide = args.use_static_guide 
test_imgs_with_guide = []
train_imgs_with_guide = []
baseDirImg = os.path.join(baseDir, 'JPEGImages', '480p')
baseDirLabel = os.path.join(baseDir, 'Annotations', '480p_split')
result_path = args.result_path #os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSMN')
for name in val_seq_names:
    test_frames = sorted(os.listdir(os.path.join(baseDirImg, name)))
    label_fds = os.listdir(os.path.join(baseDirLabel, name))
    for label_id in label_fds:
        test_imgs_with_guide += [(os.path.join(baseDirImg, name, '00000.jpg'), 
                os.path.join(baseDirLabel, name, label_id, '00000.png'),
                os.path.join(baseDirImg, name, '00001.jpg'))]
        
        if not use_static_guide:
            # use the guide image predicted from previous frame
            test_imgs_with_guide += [(os.path.join(baseDirImg, name, prev_frame), 
                os.path.join(result_path, name, label_id, prev_frame[:-4]+'.png'),
                    os.path.join(baseDirImg, name, frame)) 
                    for prev_frame, frame in zip(test_frames[1:-1], test_frames[2:])]
        else:
            # use the static visual guide image and predicted spatial guide image of previous frame
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
            # use the ground truth guide image from previous frame
            train_imgs_with_guide += [(os.path.join(baseDirImg, name, prev_frame),
                os.path.join(baseDirLabel, name, label_id, prev_frame[:-4] + '.png'),
                    os.path.join(baseDirImg, name, frame),
                    os.path.join(baseDirLabel, name, label_id, frame[:-4] + '.png')) 
                    for prev_frame, frame in zip(train_frames[:-1], train_frames[1:])]
        else:
            # use the first fram as visual guide and ground truth of previous frame as spatial guide
            train_imgs_with_guide += [(os.path.join(baseDirImg, name, '00000.jpg'),
                os.path.join(baseDirLabel, name, label_id, '00000.png'),
                os.path.join(baseDirLabel, name, label_id, prev_frame[:-4] + '.png'),
                os.path.join(baseDirImg, name, frame),
                os.path.join(baseDirLabel, name, label_id, frame[:-4] + '.png'))
                for prev_frame, frame in zip(train_frames[:-1], train_frames[1:])]

# Define Dataset
dataset = Dataset(train_imgs_with_guide, test_imgs_with_guide, data_aug=True, data_aug_scales=args.data_aug_scales)
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(args.gpu_id)):
        if args.is_training:
            max_training_iters = args.training_iters
            save_step = args.save_iters
            display_step = args.display_iters
            logs_path = args.model_save_path
            global_step = tf.Variable(0, name='global_step', trainable=False)
            osmn.train_finetune(dataset, args, args.src_model_path, None, args.learning_rate, logs_path, max_training_iters,
                             save_step, display_step, global_step, 
                             batch_size = args.batch_size,
                             iter_mean_grad=1, resume_training=args.resume_training, ckpt_name='osmn')

# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(args.gpu_id)):
        if args.is_training:
            checkpoint_path = os.path.join(logs_path, 'osmn.ckpt-'+str(max_training_iters))
        else:
            checkpoint_path = args.src_model_path    
        osmn.test(dataset, args, checkpoint_path, result_path, batch_size=1)
