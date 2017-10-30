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
from dataset_coco_vs import Dataset

def add_arguments(parser):
    group = parser.add_argument_group(title='Paths Arguments')
    group.add_argument(
            '--data_path',
            type=str,
            required=False,
            default='/raid/ljyang/data/MS_COCO')
    group.add_argument(
            '--src_model_path',
            type=str,
            required=False,
            default='models/vgg_16.ckpt')
    group.add_argument(
            '--seg_model_path',
            type=str,
            required=False,
            default='models/vgg_16.ckpt')
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
    ## masktrack params
    group.add_argument(
            '--aligned_size',
            type=list,
            required=False,
            default=None)
    group.add_argument(
            '--masktrack',
            required=False,
            action='store_true',
            default=False)
    group.add_argument(
            '--train_seg',
            required=False,
            action='store_true',
            default=False)
    group.add_argument(
            '--no_visual_modulator',
            dest='use_visual_modulator',
            action = 'store_false',
            default=True)
    group.add_argument(
            '--loss_normalize',
            action = 'store_true',
            default=False)
    group = parser.add_argument_group(title='Data Argument')
    group.add_argument(
            '--input_size',
            type=int,
            required = False,
            default=400)
    group.add_argument(
            '--data_aug_scales',
            nargs='+', type=float,
            required=False,
            default=[1])
    group.add_argument(
            '--no_visual_guide_mask',
            dest='visual_guide_mask',
            required=False,
            action='store_false',
            default=True)
    group.add_argument(
            '--sp_guide_random_blank',
            required=False,
            action='store_true',
            default=False)
            
    group.add_argument(
            '--adaptive_crop_testing',
            required=False,
            action='store_true',
            default=False,
            help="""
                use adaptive croppping around spatial guide to do testing
                """)
    group.add_argument(
            '--batch_size',
            type=int,
            required=False,
            default=6)

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
            default=100000)
    group.add_argument(
            '--save_iters',
            type=int,
            required=False,
            default=1000)
    group.add_argument(
            '--learning_rate',
            type=float,
            required=False,
            default=1e-5)
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
baseDir = args.data_path
# User defined parameters
train_path = os.path.join(baseDir, 'train2017/{:012d}.jpg')
val_path = os.path.join(baseDir, 'val2017/{:012d}.jpg')
train_file = os.path.join(baseDir, 'annotations/instances_train2017.json')
val_file = os.path.join(baseDir, 'annotations/instances_val2017.json')
print args
sys.stdout.flush()
if args.masktrack:
    import masktrack as osmn
# Define Dataset
dataset = Dataset(train_file, val_file, train_path, val_path, 
        visual_guide_mask=args.visual_guide_mask, sp_guide_random_blank=args.sp_guide_random_blank,
        use_original_mask = args.masktrack, data_aug=True, input_size=args.input_size, data_aug_scales=args.data_aug_scales)
# Train parameters
logs_path = args.model_save_path
max_training_iters = args.training_iters
learning_rate = args.learning_rate
save_step = args.save_iters
display_step = args.display_iters
batch_size = args.batch_size
src_model_path = args.src_model_path
seg_model_path = args.seg_model_path
resume_training = args.resume_training
result_path = args.result_path
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
            osmn.train_finetune(dataset, args, src_model_path, seg_model_path, learning_rate, logs_path, max_training_iters,
                                 save_step, display_step, global_step, batch_size = batch_size, config=config, 
                                 iter_mean_grad=1, use_image_summary=use_image_summary, resume_training=resume_training, ckpt_name='osmn')

# Test the network
with tf.Graph().as_default():
    with tf.device('/gpu:' + str(args.gpu_id)):
        if not args.only_testing:
            checkpoint_path = os.path.join(logs_path, 'osmn.ckpt-'+str(max_training_iters))
        else:
            checkpoint_path = src_model_path
        osmn.test(dataset, args, checkpoint_path, result_path, config=config, batch_size = batch_size)
