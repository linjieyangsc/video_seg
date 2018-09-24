"""
Train / validation script
One-Shot Modulation Netowrk
"""
import os
import sys
from PIL import Image
import numpy as np
import json
import tensorflow as tf
slim = tf.contrib.slim
import argparse
import osmn
from dataset_davis import Dataset
import random
from util import get_mask_bbox
import cPickle as pickle
import common_args 
def add_arguments(parser):
    group = parser.add_argument_group('Additional params')
    group.add_argument(
            '--data_path',
            type=str,
            required=False,
            default='/raid/ljyang/data/LSVOS',
            help='Path to LSVOS dataset')
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
            help='Source model path, could be a model pretrained on MS COCO')
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
            help='Which split to use for testing? val, train or test')
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
            default=[1],
            help='Image scales to be used by data augmentation')
    group.add_argument(
            '--use_cached_list',
            action='store_true',
            default=False,
            help='Use cache train/test list')
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
train_path = os.path.join(baseDir, 'all_train_seqs.json')
val_path = os.path.join(baseDir, 'all_%s_seqs.json' % (args.test_split))

with open(val_path, 'r') as f:
    val_seqs = json.load(f)
with open(train_path, 'r') as f:
    train_seqs = json.load(f)
randomize_guide = args.randomize_guide
test_imgs_with_guide = []
train_imgs_with_guide = []
cache_file = os.path.join('cache','lsvos_train_val_list.pkl')
if  args.use_cached_list and os.path.exists(cache_file):
    print "restore data list from cached file"
    test_imgs_with_guide, train_imgs_with_guide = pickle.load(open(cache_file,'rb'))
else:
    print "generating data list..."
    resDirLabel = args.result_path
    for seq in val_seqs:
        vid_frames = seq['frames']
        vid_anno_path = seq['anno_path']
        vid_image_path = seq['image_path']
        vid_id = seq['vid']
        for label_id, frames in vid_frames.iteritems():
            label_id = int(label_id)
            res_fd = os.path.join(vid_id, str(label_id))
            # each sample: visual guide image, visual guide mask, spatial guide mask, input image
            test_imgs_with_guide += [(os.path.join(vid_image_path, frames[0].split('.')[0] + '.jpg'), 
                    os.path.join(vid_anno_path, frames[0]),
                    None, None, label_id, res_fd)]
            # reuse the visual modulation parameters and use predicted spatial guide image of previous frame
            
            test_imgs_with_guide += [(None, None, os.path.join(vid_anno_path, frames[0]),
                    os.path.join(vid_image_path, frames[1].split('.')[0]+'.jpg'), label_id, res_fd)]
            test_imgs_with_guide += [(None, None,
                    os.path.join(resDirLabel, res_fd, prev_frame),
                    os.path.join(vid_image_path, frame.split('.')[0]+'.jpg'), 0, res_fd)
                    for prev_frame, frame in zip(frames[1:-1], frames[2:])]
                        
    for seq in train_seqs:
        vid_frames = seq['frames']
        vid_anno_path = seq['anno_path']
        vid_image_path = seq['image_path']
        vid_id = seq['vid']
        for label_id,frames in vid_frames.iteritems():
            # each sample: visual guide image, visual guide mask, spatial guide mask, input image, ground truth mask
            label_id = int(label_id)
            if randomize_guide:
                # filter images to get good quality visual guide images
                valid_label_idx = []
                for frame in frames:
                    label = Image.open(os.path.join(vid_anno_path, frame))
                    label = label.resize(args.im_size)
                    label_data = (np.array(label) == label_id)
                    bbox = get_mask_bbox(label_data, border_pixels=0)
                    if np.sum(label_data) > label_data.size * args.label_valid_ratio and \
                            np.sum(label_data) > (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) * args.bbox_valid_ratio:
                        valid_label_idx.append(frame[:-4])
                if len(valid_label_idx) > 0:
                    # randomly select guide image for each frame
                    random_guide_idx = np.random.randint(0, len(valid_label_idx),(len(frames)))
                else:
                    # default to use the first frame
                    valid_label_idx = [frames[0][:-4]]
                    random_guide_idx = np.zeros((len(frames)), dtype=np.int32)
                # use random frame as visual guide and ground truth of previous frame as spatial guide
                train_imgs_with_guide += [(os.path.join(vid_image_path, valid_label_idx[guide_id]+'.jpg'),
                    os.path.join(vid_anno_path, valid_label_idx[guide_id]+'.png'),
                    os.path.join(vid_anno_path, prev_frame),
                    os.path.join(vid_image_path, frame[:-4]+'.jpg'),
                    os.path.join(vid_anno_path, frame), label_id)
                    for prev_frame, frame, guide_id in zip(frames[:-1], frames[1:], random_guide_idx[1:])]
                
            else:
                # use the first fram as visual guide and ground truth of previous frame as spatial guide
                train_imgs_with_guide += [(os.path.join(vid_image_path, frames[0][:-4]+'.jpg'),
                    os.path.join(vid_anno_path, frames[0]),
                    os.path.join(vid_anno_path, prev_frame),
                    os.path.join(vid_image_path, frame[:-4] + '.jpg'),
                    os.path.join(vid_anno_path, frame), label_id)
                    for prev_frame, frame in zip(frames[:-1], frames[1:])]
    if args.use_cached_list: 
        with open(cache_file,'wb') as f:
            pickle.dump((test_imgs_with_guide, train_imgs_with_guide), f)
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
