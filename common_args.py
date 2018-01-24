import argparse

def add_arguments(parser):
    group = parser.add_argument_group(title='Paths Arguments')
    group.add_argument(
            '--result_path',
            type=str,
            required=True,
            default='',
            help='Path to save predicted results')
    group.add_argument(
            '--model_save_path',
            type=str,
            required=False,
            default='',
            help='Path to save models')
    group = parser.add_argument_group(title='Model Arguments')
    group.add_argument(
            '--mod_early_conv',
            required=False,
            action='store_true',
            default=False,
            help='Whether to modulate conv1 and conv2 in VGG model')
    group.add_argument(
            '--no_spatial_modulator',
            required=False,
            dest='use_spatial_modulator',
            action='store_false',
            default=True,
            help='Not using spatial modulator')
    group.add_argument(
            '--no_visual_modulator',
            required=False,
            dest='use_visual_modulator',
            action='store_false',
            default=True,
            help='Not using visual modulator')
    group.add_argument(
            '--fix_bn',
            required=False,
            action='store_true',
            default=False,
            help='Fix batch norm parameters, set this for finetuning on DAVIS')
    ## masktrack params
    # set aligned size to 865, 481 for testing MaskTrack on davis
    group.add_argument(
            '--aligned_size',
            type=int, nargs=2,
            required=False,
            default=None,
            help='Used by MaskTrack model due the padding difference between Caffe and TF.')

    group.add_argument(
            '--fix_seg',
            dest='train_seg',
            required=False,
            action='store_false',
            default=True,
            help='Whether to fix segmentation model in training')
    group.add_argument(
            '--bbox_sup',
            required=False,
            action='store_true',
            default=False,
            help='Only use bbox as supervision, visual guide will be a cropped image with bbox.')
    group.add_argument(
            '--use_original_mask',
            required=False,
            action='store_true',
            default=False,
            help='Whether to use original mask as input. Used by MaskTrack')
    group.add_argument(
            '--base_model',
            required=False,
            type=str,
            choices=['osvos','masktrack','deeplab'],
            default="osvos",
            help='Pick one model type from: osvos, masktrack, deeplab')
    group = parser.add_argument_group(title='Data Argument')
    group.add_argument(
            '--crf_postprocessing',
            required=False,
            action='store_true',
            default=False,
            help='whether or not use crf postprocessing')
    group.add_argument(
            '--adaptive_crop_testing',
            required=False,
            action='store_true',
            default=False,
            help='Use adaptive croppping around spatial guide to do testing')
    group.add_argument(
            '--random_crop_ratio',
            required=False,
            default=0.0,
            type=float,
            help='Random crop ratio on the input image')
    group.add_argument(
            '--vg_keep_aspect_ratio',
            required=False,
            action = 'store_true',
            help='Whether to keep aspect ratio when resizing visual guide')
    group.add_argument(
            '--vg_color_aug',
            required=False,
            action ='store_true',
            default=False,
            help='Whether to do color augmentation on visual guide')
    group.add_argument(
            '--vg_random_crop_ratio',
            required=False,
            default=0.0,
            type=float,
            help='Random crop ratio on visual guide')
    group.add_argument(
            '--vg_random_rotate_angle',
            required=False,
            default=0,
            type=int,
            help='Random rotation angle on visual guide')
    group.add_argument(
            '--sg_center_perturb_ratio',
            required=False,
            default=0.2,
            type=float,
            help='Center perturbation ratio on spatial guide')
    group.add_argument(
            '--sg_std_perturb_ratio',
            required=False,
            default=0.4,
            type=float,
            help='Size perturbation ratio on spatial guide')
    group.add_argument(
            '--batch_size',
            type=int,
            required=False,
            default=8,
            help='Batch size in training')
    group.add_argument(
            '--save_score',
            required=False,
            action='store_true',
            default=False,
            help='Whether to save prediction score. Need to be set for DAVIS 2017 for prediction combination.')
    group = parser.add_argument_group(title='Running Arguments')
    group.add_argument(
            '--gpu_id',
            type=int,
            required=False,
            default=0,
            help='GPU id to use')
    group.add_argument(
            '--training_iters',
            type=int,
            required=False,
            default=100000,
            help='Training iterations')
    group.add_argument(
            '--save_iters',
            type=int,
            required=False,
            default=1000,
            help='Save model per this number of iterations')
    group.add_argument(
            '--learning_rate',
            type=float,
            required=False,
            default=1e-5,
            help='Learning rate in training')
    group.add_argument(
            '--display_iters',
            type=int,
            required=False,
            default=20,
            help='Show summary statistics every this number of iterations')
    group.add_argument(
            '--use_image_summary',
            required=False,
            action='store_true',
            default=False,
            help='Add ongoing image results to tensorboard')
    group.add_argument(
            '--only_testing',
            required=False,
            action='store_true',
            default=False,
            help='Only testing the model, no training')
    group.add_argument(
            '--restart_training',
            dest='resume_training',
            required=False,
            action='store_false',
            default=True,
            help='Restart model training, ignore existed model files in the model folder')
