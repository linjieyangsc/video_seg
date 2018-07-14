"""
Train / test script
One-Shot Modulation Netowrk with online finetuning
"""
import os
import sys
from PIL import Image
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import json
import argparse
import osmn
from dataset_davis import Dataset
import random
from util import get_mask_bbox
import common_args 
def add_arguments(parser):
    group = parser.add_argument_group('Additional params')
    group.add_argument(
            '--data_path',
            type=str,
            required=False,
            default='/raid/ljyang/data/YoutubeVOS')
    group.add_argument(
            '--seg_model_path',
            type=str,
            required=False,
            default='')
    group.add_argument(
            '--whole_model_path',
            type=str,
            required=False,
            default='')
    group.add_argument(
            '--start_id',
            type=str,
            required=False,
            default='')

    group.add_argument(
            '--test_split',
            type=str,
            required=False,
            default='valid'
            )
    group.add_argument(
            '--im_size',
            nargs=2, type=int,
            required = False,
            default=[448, 256],
            help='Input image size')
    group.add_argument(
            '--data_aug_scales',
            type=float, nargs='+',
            required=False,
            default=[0.8,1,1.5])
print " ".join(sys.argv[:])
parser = argparse.ArgumentParser()
common_args.add_arguments(parser)
add_arguments(parser)
args = parser.parse_args()
print args
sys.stdout.flush()
baseDir = args.data_path
random.seed(1234)
# User defined parameters
val_path = os.path.join(baseDir, args.test_split, 'meta.json' )
## default config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

with open(val_path, 'r') as f:
    val_seqs = json.load(f)['videos']
resDirLabel = args.result_path
if args.start_id != '':
    idx = [s['vid'] for s in val_seqs].index(args.start_id)
    val_seqs = val_seqs[idx:]
for vid_id, seq in val_seqs.iteritems():
    vid_frames = seq['objects']
    vid_anno_path = os.path.join(baseDir, args.test_split, 'Annotations', vid_id)
    vid_image_path = os.path.join(baseDir, args.test_split, 'JPEGImages', vid_id)
    for label_id, obj_info in vid_frames.iteritems():
        frames = obj_info['frames']
        # train on first frame test on whole sequence
        res_fd = os.path.join(vid_id, label_id)
        train_imgs_with_guide = [(os.path.join(vid_image_path, frames[0]+'.jpg'), 
                os.path.join(vid_anno_path, frames[0]+'.png'),
                os.path.join(vid_anno_path, frames[0]+'.png'),
                os.path.join(vid_image_path, frames[0]+'.jpg'),
                os.path.join(vid_anno_path, frames[0])+'.png', int(label_id))]      
        test_imgs_with_guide = []
        
        # each sample: visual guide image, visual guide mask, spatial guide mask, input image
        test_imgs_with_guide += [(os.path.join(vid_image_path, frames[0] + '.jpg'), 
                os.path.join(vid_anno_path, frames[0]+'.png'),
                None, None, int(label_id), res_fd)]
        # reuse the visual modulation parameters and use predicted spatial guide image of previous frame
        
        test_imgs_with_guide += [(None, None, os.path.join(vid_anno_path, frames[0]+'.png'),
                os.path.join(vid_image_path, frames[1]+'.jpg'), int(label_id), res_fd)]
        test_imgs_with_guide += [(None, None,
                os.path.join(resDirLabel, res_fd, prev_frame+'.png'),
                os.path.join(vid_image_path, frame+'.jpg'), 0, res_fd)
                for prev_frame, frame in zip(frames[1:-1], frames[2:])]
        # Define Dataset
        dataset = Dataset(train_imgs_with_guide, test_imgs_with_guide, args,
                data_aug=True)
    

        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(args.gpu_id)):
                if not args.only_testing:
                    max_training_iters = args.training_iters
                    save_step = args.save_iters
                    display_step = args.display_iters
                    logs_path = os.path.join(args.model_save_path, vid_id, label_id)
                    global_step = tf.Variable(0, name='global_step', trainable=False)
                    osmn.train_finetune(dataset, args, args.learning_rate, logs_path, max_training_iters,
                                     save_step, display_step, global_step, 
                                     batch_size = 1, config=config,
                                     iter_mean_grad=1, use_image_summary = args.use_image_summary, resume_training=args.resume_training, ckpt_name='osmn')

        # Test the network
        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(args.gpu_id)):
                if not args.only_testing:
                    checkpoint_path = os.path.join(logs_path, 'osmn.ckpt-'+str(max_training_iters))
                else:
                    checkpoint_path = args.whole_model_path    
                osmn.test(dataset, args, checkpoint_path, args.result_path, config=config, batch_size=1)
