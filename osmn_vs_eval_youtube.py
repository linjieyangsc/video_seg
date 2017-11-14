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
import osmn_vs as osmn
from dataset_davis_vs import Dataset
import common_args
def add_arguments(parser):
    group = parser.add_argument_group('Additional params')
    group.add_argument(
            '--data_path',
            type=str,
            required=False,
            default='/raid/ljyang/data/youtube_masks')
    group.add_argument(
            '--src_model_path',
            type=str,
            required=False,
            default='')
    group.add_argument(
            '--seg_model_path',
            type=str,
            required=False,
            default='')
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
use_static_guide = args.use_static_guide 
test_imgs_with_guide = []
baseDirImg = os.path.join(baseDir, 'Images')
baseDirLabel = os.path.join(baseDir, 'Labels')
result_path = args.result_path #os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSMN')
guideDirLabel = result_path
for name in val_seq_names:
    print name
    test_frames = sorted(os.listdir(os.path.join(baseDirImg, name)))
    if len(test_frames) < 2:
        continue
    test_imgs_with_guide += [(os.path.join(baseDirImg, name, test_frames[0]), 
        os.path.join(baseDirLabel, name, test_frames[0][:-4]+'.png'),
            os.path.join(baseDirImg, name, test_frames[1]))]
    
    if not use_static_guide:
        # use the guide image predicted from previous frame
        test_imgs_with_guide += [(os.path.join(baseDirImg, name, prev_frame), 
            os.path.join(guideDirLabel, name, prev_frame[:-4]+'.png'),
                os.path.join(baseDirImg, name, frame)) 
                for prev_frame, frame in zip(test_frames[1:-1], test_frames[2:])]
    else:
        # use the static visual guide image and predicted spatial guide image of previous frame
        test_imgs_with_guide += [(os.path.join(baseDirImg, name, test_frames[0]),
            os.path.join(baseDirLabel, name, test_frames[0][:-4]+'.png'),
                os.path.join(guideDirLabel, name, prev_frame[:-4] +'.png'),
                os.path.join(baseDirImg, name, frame))
                for prev_frame, frame in zip(test_frames[1:-1], test_frames[2:])]
                    
# Define Dataset
im_size = [640, 360]
dataset = Dataset([], test_imgs_with_guide, 
        im_size = im_size,
        multiclass = False,
        adaptive_crop_testing = args.adaptive_crop_testing,
        use_original_mask = args.masktrack,
        crf_preprocessing = args.crf_preprocessing)
if args.masktrack:
    import masktrack as osmn
    
## default config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9


# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(args.gpu_id)):
        checkpoint_path = args.src_model_path    
        osmn.test(dataset, args, checkpoint_path, result_path, config=config, batch_size=1)
