"""
One-Shot Modulation Netowrk
Training on MS-COCO
"""
import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import argparse
import osmn
from dataset_coco import Dataset
import common_args
def add_arguments(parser):
    group = parser.add_argument_group('Additional params')
    group.add_argument(
            '--data_path',
            type=str,
            required=False,
            default='/raid/ljyang/data/MS_COCO',
            help='Path to the MS COCO dataset')
    group.add_argument(
            '--vis_mod_model_path',
            type=str,
            required=False,
            default='models/vgg_16.ckpt',
            help='Model to initialize visual modulator')
    group.add_argument(
            '--seg_model_path',
            type=str,
            required=False,
            default='models/vgg_16.ckpt',
            help='Model to initialize segmentation model')

    group.add_argument(
            '--whole_model_path',
            type=str,
            required=False,
            default='',
            help='Model to initialize the whole model')
    group.add_argument(
            '--im_size',
            nargs=2, type=int,
            required = False,
            default=[400, 400],
            help='Input image size')
    group.add_argument(
            '--data_aug_scales',
            nargs='+', type=float,
            required=False,
            default=[0.8, 1, 1.2],
            help='Image scales for data augmentation')

print " ".join(sys.argv[:])

parser = argparse.ArgumentParser()
common_args.add_arguments(parser)
add_arguments(parser)
args = parser.parse_args()
baseDir = args.data_path
# User defined parameters
train_path = os.path.join(baseDir, 'train2017/{:012d}.jpg')
val_path = os.path.join(baseDir, 'val2017/{:012d}.jpg')
train_file = os.path.join(baseDir, 'annotations/instances_train2017.json')
val_file = os.path.join(baseDir, 'annotations/instances_val2017.json')
print args
sys.stdout.flush()
# Define Dataset
dataset = Dataset(train_file, val_file, train_path, val_path, args, data_aug=True)
# Train parameters
logs_path = args.model_save_path
max_training_iters = args.training_iters
learning_rate = args.learning_rate
save_step = args.save_iters
display_step = args.display_iters
batch_size = args.batch_size
resume_training = args.resume_training
use_image_summary = args.use_image_summary
## default config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
if not args.only_testing:
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(args.gpu_id)):
            global_step = tf.Variable(0, name='global_step', trainable=False)
            osmn.train_finetune(dataset, args, learning_rate, logs_path, max_training_iters,
                                 save_step, display_step, global_step, batch_size = batch_size, config=config, 
                                 iter_mean_grad=1, use_image_summary=use_image_summary, resume_training=resume_training, ckpt_name='osmn')

