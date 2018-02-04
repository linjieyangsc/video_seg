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
            default='/raid/ljyang/data/DAVIS')
    group.add_argument(
            '--whole_model_path',
            type=str,
            required=False,
            default='')
    group.add_argument(
            '--data_version',
            type=int,
            required=False,
            default=2017,
            help="""
                which DAVIS version to use? 2016 or 2017
                """)
    group.add_argument(
            '--test_split',
            type=str,
            required=False,
            default='val'
            )
    group.add_argument(
            '--im_size',
            nargs=2, type=int,
            required = False,
            default=[854, 480],
            help='Input image size')
    group.add_argument(
            '--data_aug_scales',
            type=float, nargs='+',
            required=False,
            default=[0.5,0.8,1])
print " ".join(sys.argv[:])
parser = argparse.ArgumentParser()
common_args.add_arguments(parser)
add_arguments(parser)
args = parser.parse_args()
print args
sys.stdout.flush()
baseDir = args.data_path
data_version =args.data_version
random.seed(1234)
# User defined parameters
val_path = os.path.join(baseDir, 'ImageSets/%d/%s.txt' % (data_version, args.test_split))
## default config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

with open(val_path, 'r') as f:
    val_seq_names = [line.strip() for line in f]
baseDirImg = os.path.join(baseDir, 'JPEGImages', 'Full-Resolution')
label_fd = '480p_split' if data_version==2017 else '480p_all'
baseDirLabel = os.path.join(baseDir, 'Annotations', label_fd)
resDirLabel = args.result_path
for name in val_seq_names:
    test_frames = sorted(os.listdir(os.path.join(baseDirImg, name)))
    label_fds = os.listdir(os.path.join(baseDirLabel, name)) if data_version == 2017 else \
            ['']
    for label_id in label_fds:
    # train on first frame test on whole sequence
    
        train_imgs_with_guide = [(os.path.join(baseDirImg, name, '00000.jpg'), 
                os.path.join(baseDirLabel, name, label_id, '00000.png'),
                os.path.join(baseDirImg, name, '00000.jpg'),
                os.path.join(baseDirLabel, name, label_id, '00000.png'))]
        
        test_imgs_with_guide = []
        test_imgs_with_guide += [(os.path.join(baseDirImg, name, '00000.jpg'), 
                os.path.join(baseDirLabel, name, label_id, '00000.png'),
                None, None)]
        
        test_imgs_with_guide += [(None, None,
                os.path.join(baseDirLabel, name, label_id, '00000.png'),
                os.path.join(baseDirImg, name, '00001.jpg'))]
        test_imgs_with_guide += [(None, None,
                os.path.join(resDirLabel, name, label_id, prev_frame[:-4] +'.png'),
                os.path.join(baseDirImg, name, frame))
                for prev_frame, frame in zip(test_frames[1:-1], test_frames[2:])]
                    
        # Define Dataset
        dataset = Dataset(train_imgs_with_guide, test_imgs_with_guide, args,
                data_aug=True)
    

        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(args.gpu_id)):
                if not args.only_testing:
                    max_training_iters = args.training_iters
                    save_step = args.save_iters
                    display_step = args.display_iters
                    logs_path = os.path.join(args.model_save_path, name, label_id)
                    global_step = tf.Variable(0, name='global_step', trainable=False)
                    osmn.train_finetune(dataset, args, args.learning_rate, logs_path, max_training_iters,
                                     save_step, display_step, global_step, 
                                     batch_size = args.batch_size, config=config,
                                     iter_mean_grad=1, use_image_summary = args.use_image_summary, resume_training=args.resume_training, ckpt_name='osmn')

        # Test the network
        with tf.Graph().as_default():
            with tf.device('/gpu:' + str(args.gpu_id)):
                if not args.only_testing:
                    checkpoint_path = os.path.join(logs_path, 'osmn.ckpt-'+str(max_training_iters))
                else:
                    checkpoint_path = args.whole_model_path    
                osmn.test(dataset, args, checkpoint_path, args.result_path, config=config, batch_size=1)
