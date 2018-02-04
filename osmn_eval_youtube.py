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
import argparse
import osmn
from dataset_davis import Dataset
import common_args
def add_arguments(parser):
    group = parser.add_argument_group('Additional params')
    group.add_argument(
            '--data_path',
            type=str,
            required=False,
            default='/raid/ljyang/data/youtube_masks')
    group.add_argument(
            '--whole_model_path',
            type=str,
            required=True,
            default='')
    group.add_argument(
            '--im_size',
            nargs=2, type=int,
            required = False,
            default=[640, 360],
            help='Input image size')
    group.add_argument(
            '--data_aug_scales',
            type=float, nargs='+',
            required=False,
            default=[1],
            help='Image scales to be used by data augmentation')
print " ".join(sys.argv[:])
parser = argparse.ArgumentParser()
common_args.add_arguments(parser)
add_arguments(parser)
args = parser.parse_args()
print args
sys.stdout.flush()
baseDir = args.data_path

# User defined parameters
val_seq_path = os.path.join(baseDir, 'all.txt')
with open(val_seq_path) as f:
    val_seq_names = [line.strip() for line in f]
test_imgs_with_guide = []
baseDirImg = os.path.join(baseDir, 'Images')
baseDirLabel = os.path.join(baseDir, 'Labels')
resDirLabel = args.result_path
for name in val_seq_names:
    print name
    test_frames = sorted(os.listdir(os.path.join(baseDirImg, name)))
    if len(test_frames) < 2:
        continue
    test_imgs_with_guide += [(os.path.join(baseDirImg, name, test_frames[0]), 
        os.path.join(baseDirLabel, name, test_frames[0][:-4]+'.png'),
            None, None)]
    test_imgs_with_guide += [(None, None,
        os.path.join(baseDirLabel, name, test_frames[0][:-4]+'.png'),
            os.path.join(baseDirImg, name, test_frames[1]))]
    
    test_imgs_with_guide += [(None, None,
            os.path.join(resDirLabel, name, prev_frame[:-4] +'.png'),
            os.path.join(baseDirImg, name, frame))
            for prev_frame, frame in zip(test_frames[1:-1], test_frames[2:])]
                    
# Define Dataset
dataset = Dataset([], test_imgs_with_guide, 
        args)
    
## default config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9


# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(args.gpu_id)):
        checkpoint_path = args.whole_model_path    
        osmn.test(dataset, args, checkpoint_path, args.result_path, config=config, batch_size=1)
