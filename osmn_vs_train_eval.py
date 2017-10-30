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
            required=False,
            default='')
    group.add_argument(
            '--seg_model_path',
            type=str,
            required=False,
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
    group = parser.add_argument_group(title='Model Arguments')
    group.add_argument(
            '--mod_last_conv',
            required=False,
            action='store_true',
            default=False)
    group.add_argument(
            '--mod_early_conv',
            required=False,
            action='store_true',
            default=False)
    group.add_argument(
            '--trimmed_mod',
            required=False,
            action = 'store_true',
            default=False)
    group.add_argument(
            '--orig_gb',
            required=False,
            action='store_true',
            default=False)
    group.add_argument(
            '--sp_late_fusion',
            required=False,
            action='store_true',
            default=False)
    group.add_argument(
            '--spatial_mod_use_bn',
            required=False,
            action='store_true',
            default=False)
    group.add_argument(
            '--no_visual_modulator',
            required=False,
            dest='use_visual_modulator',
            action='store_false',
            default=True)
    group.add_argument(
            '--loss_normalize',
            required=False,
            action='store_true',
            default=False)
    ## masktrack params
    group.add_argument(
            '--aligned_size',
            type=list,
            required=False,
            default=[865, 481])
    group.add_argument(
            '--train_seg',
            required=False,
            action='store_true',
            default=False)
    group.add_argument(
            '--masktrack',
            required=False,
            action='store_true',
            default=False)
    group = parser.add_argument_group(title='Data Argument')
    group.add_argument(
            '--use_prev_guide',
            dest='use_static_guide',
            required=False,
            action='store_false',
            default=True,
            help="""
                only use the first frame as visual guide or use the previous frame as visual guide
                """)
    group.add_argument(
            '--crf_preprocessing',
            dest='crf_preprocessing',
            required=False,
            action='store_true',
            default=False,
            help="""
                whether or not use crf preprocessing for masktrack method
                """)
    group.add_argument(
            '--adaptive_crop_testing',
            required=False,
            action='store_true',
            default=False,
            help="""
                use adaptive croppping around spatial guide to do testing
                """)
    group.add_argument(
            '--data_aug_scales',
            type=list,
            required=False,
            default=[0.5,0.8,1])
    group.add_argument(
            '--no_guide_image_mask',
            dest='guide_image_mask',
            required=False,
            action='store_false',
            default=True)
    group.add_argument(
            '--sp_guide_random_blank',
            required=False,
            action='store_true',
            default=False)
    group.add_argument(
            '--batch_size',
            type=int,
            required=False,
            default=4)
    group.add_argument(
            '--save_score',
            required=False,
            action='store_true',
            default=False)
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
            default=20)
    group.add_argument(
            '--use_image_summary',
            required=False,
            action='store_true',
            default=False,
            help="""
                add valdiation image results to tensorboard
                """)
    group.add_argument(
            '--only_testing',
            required=False,
            action='store_true',
            default=False,
            help="""\
                is it training or testing?
                """)
    group.add_argument(
            '--restart_training',
            dest='resume_training',
            required=False,
            action='store_false',
            default=True)
print " ".join(sys.argv[:])
parser = argparse.ArgumentParser()
add_arguments(parser)
args = parser.parse_args()
print args
sys.stdout.flush()
baseDir = args.data_path
data_version =args.data_version

# User defined parameters
train_path = os.path.join(baseDir, 'ImageSets/%d/train.txt' % data_version)
val_path = os.path.join(baseDir, 'ImageSets/%d/%s.txt' % (data_version, args.test_split))

with open(val_path, 'r') as f:
    val_seq_names = [line.strip() for line in f]
with open(train_path, 'r') as f:
    train_seq_names = [line.strip() for line in f]
use_static_guide = args.use_static_guide 
test_imgs_with_guide = []
train_imgs_with_guide = []
baseDirImg = os.path.join(baseDir, 'JPEGImages', '480p')
label_fd = '480p_split' if data_version==2017 else '480p_all'
baseDirLabel = os.path.join(baseDir, 'Annotations', label_fd)
result_path = args.result_path #os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSMN')
guideDirLabel = result_path
for name in val_seq_names:
    test_frames = sorted(os.listdir(os.path.join(baseDirImg, name)))
    label_fds = os.listdir(os.path.join(baseDirLabel, name)) if data_version == 2017 else \
            ['']
    for label_id in label_fds:
        test_imgs_with_guide += [(os.path.join(baseDirImg, name, '00000.jpg'), 
                os.path.join(baseDirLabel, name, label_id, '00000.png'),
                os.path.join(baseDirImg, name, '00000.jpg'))]
        
        test_imgs_with_guide += [(os.path.join(baseDirImg, name, '00000.jpg'), 
                os.path.join(baseDirLabel, name, label_id, '00000.png'),
                os.path.join(baseDirImg, name, '00001.jpg'))]
        if not use_static_guide:
            # use the guide image predicted from previous frame
            test_imgs_with_guide += [(os.path.join(baseDirImg, name, prev_frame), 
                os.path.join(guideDirLabel, name, label_id, prev_frame[:-4]+'.png'),
                    os.path.join(baseDirImg, name, frame)) 
                    for prev_frame, frame in zip(test_frames[1:-1], test_frames[2:])]
        else:
            # use the static visual guide image and predicted spatial guide image of previous frame
            test_imgs_with_guide += [(os.path.join(baseDirImg, name, '00000.jpg'),
                    os.path.join(baseDirLabel, name, label_id, '00000.png'),
                    os.path.join(guideDirLabel, name, label_id, prev_frame[:-4] +'.png'),
                    os.path.join(baseDirImg, name, frame))
                    for prev_frame, frame in zip(test_frames[1:-1], test_frames[2:])]
                    
for name in train_seq_names:
    train_frames = sorted(os.listdir(os.path.join(baseDirImg, name)))
    label_fds = os.listdir(os.path.join(baseDirLabel, name)) if data_version == 2017 else \
            [os.path.join(baseDirLabel, name)]
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
multiclass = (data_version == 2017)
dataset = Dataset(train_imgs_with_guide, test_imgs_with_guide, 
        multiclass = multiclass,
        adaptive_crop_testing = args.adaptive_crop_testing,
        use_original_mask = args.masktrack,
        crf_preprocessing = args.crf_preprocessing,
        sp_guide_random_blank=args.sp_guide_random_blank, 
        guide_image_mask=args.guide_image_mask, 
        data_aug=True, data_aug_scales=args.data_aug_scales)
if args.masktrack:
    import masktrack as osmn
    
## default config
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

with tf.Graph().as_default():
    with tf.device('/gpu:' + str(args.gpu_id)):
        if not args.only_testing:
            max_training_iters = args.training_iters
            save_step = args.save_iters
            display_step = args.display_iters
            logs_path = args.model_save_path
            global_step = tf.Variable(0, name='global_step', trainable=False)
            osmn.train_finetune(dataset, args, args.src_model_path, args.seg_model_path, args.learning_rate, logs_path, max_training_iters,
                             save_step, display_step, global_step, 
                             batch_size = args.batch_size, config=config,
                             iter_mean_grad=1, use_image_summary = args.use_image_summary, resume_training=args.resume_training, ckpt_name='osmn')

# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(args.gpu_id)):
        if not args.only_testing:
            checkpoint_path = os.path.join(logs_path, 'osmn.ckpt-'+str(max_training_iters))
        else:
            checkpoint_path = args.src_model_path    
        osmn.test(dataset, args, checkpoint_path, result_path, config=config, batch_size=1)
