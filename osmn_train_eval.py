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
import random
from util import get_mask_bbox
import common_args 
def add_arguments(parser):
    group = parser.add_argument_group('Additional params')
    group.add_argument(
            '--data_path',
            type=str,
            required=False,
            default='/raid/ljyang/data/DAVIS',
            help='Path to DAVIS dataset')
    group.add_argument(
            '--whole_model_path',
            type=str,
            required=True,
            default='',
            help='Source model path, could be a model pretrained on MS COCO')
    group.add_argument(
            '--data_version',
            type=int,
            required=False,
            default=2017,
            help='Which DAVIS version to use? 2016 or 2017')
    group.add_argument(
            '--use_full_res',
            required=False,
            action='store_true',
            default=False,
            help='Whether to use full resolution image, can benefit when the visual guide is small and has low resolution.')
    group.add_argument(
            '--randomize_guide',
            required=False,
            action='store_true',
            default=False,
            help='Whether to use randomized visual guide, or only the first frame')
    group.add_argument(
            '--label_valid_ratio',
            type=float,
            required=False,
            default=0.003,
            help='Parameter to search for valid visual guide, see details in code')
    group.add_argument(
            '--bbox_valid_ratio',
            type=float,
            required=False,
            default=0.2,
            help='Parameter to search for valid visual guide, see details in code')
    group.add_argument(
            '--test_split',
            type=str,
            required=False,
            default='val',
            help='Which split to use for testing? val, train or trainval')
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
            default=[0.5,0.8,1],
            help='Image scales to be used by data augmentation')
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
train_path = os.path.join(baseDir, 'ImageSets/%d/train.txt' % data_version)
val_path = os.path.join(baseDir, 'ImageSets/%d/%s.txt' % (data_version, args.test_split))

with open(val_path, 'r') as f:
    val_seq_names = [line.strip() for line in f]
with open(train_path, 'r') as f:
    train_seq_names = [line.strip() for line in f]
randomize_guide = args.randomize_guide
test_imgs_with_guide = []
train_imgs_with_guide = []
if args.use_full_res:
    baseDirImg = os.path.join(baseDir, 'JPEGImages', 'Full-Resolution')
else:
    baseDirImg = os.path.join(baseDir, 'JPEGImages', '480p')
label_fd = '480p_split' if data_version==2017 else '480p_all'
baseDirLabel = os.path.join(baseDir, 'Annotations', label_fd)
resDirLabel = args.result_path
for name in val_seq_names:
    test_frames = sorted(os.listdir(os.path.join(baseDirImg, name)))
    label_fds = os.listdir(os.path.join(baseDirLabel, name)) if data_version == 2017 else \
            ['']
    for label_id in label_fds:
        # each sample: visual guide image, visual guide mask, spatial guide mask, input image
        test_imgs_with_guide += [(os.path.join(baseDirImg, name, '00000.jpg'), 
                os.path.join(baseDirLabel, name, label_id, '00000.png'),
                None, None)]
        # reuse the visual modulation parameters and use predicted spatial guide image of previous frame
        
        test_imgs_with_guide += [(None, None, os.path.join(baseDirLabel, name, label_id, '00000.png'),
                os.path.join(baseDirImg, name, '00001.jpg'))]
        test_imgs_with_guide += [(None, None,
                os.path.join(resDirLabel, name, label_id, prev_frame[:-4] +'.png'),
                os.path.join(baseDirImg, name, frame))
                for prev_frame, frame in zip(test_frames[1:-1], test_frames[2:])]
                    
for name in train_seq_names:
    train_frames = sorted(os.listdir(os.path.join(baseDirImg, name)))
    label_fds = os.listdir(os.path.join(baseDirLabel, name)) if data_version == 2017 else \
            [os.path.join(baseDirLabel, name)]
    for label_id in label_fds:
        # each sample: visual guide image, visual guide mask, spatial guide mask, input image, ground truth mask
        if randomize_guide:
            # filter images to get good quality visual guide images
            valid_label_idx = []
            nonblank_label_idx = []
            for frame in train_frames:
                label = Image.open(os.path.join(baseDirLabel, name, label_id, frame[:-4] + '.png'))
                label_data = np.array(label) > 0
                bbox = get_mask_bbox(label_data, border_pixels=0)
                if np.sum(label_data) > 0:
                    nonblank_label_idx.append(frame)
                if np.sum(label_data) > label_data.size * args.label_valid_ratio and \
                        np.sum(label_data) > (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) * args.bbox_valid_ratio:
                    valid_label_idx.append(frame[:-4])
            train_frames = nonblank_label_idx
            if len(valid_label_idx) > 0:
                # randomly select guide image for each frame
                random_guide_idx = np.random.randint(0, len(valid_label_idx),(len(train_frames)))
            else:
                # default to use the first frame
                valid_label_idx = [train_frames[0][:-4]]
                random_guide_idx = np.zeros((len(train_frames)), dtype=np.int32)
            # use random frame as visual guide and ground truth of previous frame as spatial guide
            train_imgs_with_guide += [(os.path.join(baseDirImg, name, valid_label_idx[guide_id]+'.jpg'),
                os.path.join(baseDirLabel, name, label_id, valid_label_idx[guide_id]+'.png'),
                os.path.join(baseDirLabel, name, label_id, prev_frame[:-4] + '.png'),
                os.path.join(baseDirImg, name, frame),
                os.path.join(baseDirLabel, name, label_id, frame[:-4] + '.png'))
                for prev_frame, frame, guide_id in zip(train_frames[:-1], train_frames[1:], random_guide_idx[1:])]
            
        else:
            # use the first fram as visual guide and ground truth of previous frame as spatial guide
            train_imgs_with_guide += [(os.path.join(baseDirImg, name, '00000.jpg'),
                os.path.join(baseDirLabel, name, label_id, '00000.png'),
                os.path.join(baseDirLabel, name, label_id, prev_frame[:-4] + '.png'),
                os.path.join(baseDirImg, name, frame),
                os.path.join(baseDirLabel, name, label_id, frame[:-4] + '.png'))
                for prev_frame, frame in zip(train_frames[:-1], train_frames[1:])]

# Define Dataset
dataset = Dataset(train_imgs_with_guide, test_imgs_with_guide, args,
        data_aug=True)
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
