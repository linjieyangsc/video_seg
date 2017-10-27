"""
Train / validation script
One-Shot Modulation Netowrk
"""
import os
import sys
import scipy
from PIL import Image
import numpy as np
import argparse
import caffe
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
            required=True,
            default='')
    group.add_argument(
            '--model_proto_path',
            type=str,
            required=True,
            default='')
    group.add_argument(
            '--result_path',
            type=str,
            required=True,
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
    group = parser.add_argument_group(title='Data Argument')
    group.add_argument(
            '--online_testing',
            required=False,
            action='store_true',
            default=False,
            help="""
                offline testing: use ground truth mask from previous frame as spatial guide
                online testing: use predicted mask from previous frame as spatial guide
                default to offline, should be set to offline when there is image summaries
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
            '--no_guide_image_mask',
            dest='guide_image_mask',
            required=False,
            action='store_false',
            default=True)
    group.add_argument(
            '--batch_size',
            type=int,
            required=False,
            default=4)
    group = parser.add_argument_group(title='Running Arguments')
    group.add_argument(
            '--gpu_id',
            type=int,
            required=False,
            default=0)
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
use_static_guide = True
online_testing = args.online_testing
test_imgs_with_guide = []
train_imgs_with_guide = []
baseDirImg = os.path.join(baseDir, 'JPEGImages', '480p')
label_fd = '480p_split' if data_version==2017 else '480p_all'
baseDirLabel = os.path.join(baseDir, 'Annotations', label_fd)
result_path = args.result_path #os.path.join('DAVIS', 'Results', 'Segmentations', '480p', 'OSMN')
guideDirLabel = result_path if online_testing else baseDirLabel
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
im_size = (854, 480)
input_pad_size = (865, 481)
input_start_pos = (5,0)
adaptive_crop_testing = args.adaptive_crop_testing
dataset = Dataset(train_imgs_with_guide, test_imgs_with_guide, 
        multiclass = multiclass,
        adaptive_crop_testing = adaptive_crop_testing,
        use_original_mask = True,
        guide_image_mask=False, 
        im_size = im_size,
        data_aug=False)
# Test the network
batch_size = 1 # online testing only use batch size 1
caffe.set_mode_gpu()
caffe.set_device(args.gpu_id)
net = caffe.Net(args.model_proto_path, args.src_model_path, caffe.TEST)
net.blobs["data"].reshape(batch_size, 4, input_pad_size[1], input_pad_size[0])
if not os.path.exists(result_path):
    os.makedirs(result_path)
for frame in range(0, dataset.get_test_size(), batch_size):
    guide_images, mask_images, images, image_paths = dataset.next_batch(batch_size, 'test')
    save_names = [name.split('.')[0] + '.png' for name in image_paths]
    combined_im = np.concatenate((images, mask_images), axis=3)
    combined_im = combined_im.transpose((0,3,1,2))
    net.blobs["data"].data[0,:,input_start_pos[1]:input_start_pos[1]+im_size[1],
            input_start_pos[0]:input_start_pos[0]+im_size[0]] = combined_im
    net.forward()
    res = net.blobs["prob"].data
    crop_h = (res.shape[2] - im_size[1]) / 2
    crop_w = (res.shape[3] - im_size[0]) / 2
    res = res[:,:,crop_h:crop_h + im_size[1], crop_w:crop_w + im_size[0]]
    
    labels = res.argmax(1)[0]
    score = res[0,1,:,:]
    #score = np.array(Image.fromarray(res[0,:,:,1],mode='F').resize(im_size, Image.BILINEAR))
    #labels = np.array(Image.fromarray(labels.astype(np.uint8)).resize(im_size, Image.NEAREST))
    print 'Saving ' + os.path.join(result_path,save_names[0])
    if len(save_names[0].split('/')) > 1:
        save_path = os.path.join(result_path, *(save_names[0].split('/')[:-1]))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    scipy.misc.imsave(os.path.join(result_path, save_names[0]), labels.astype(np.float32))
    curr_score_name = save_names[0][:-4]
    #print 'Saving ' + os.path.join(result_path, curr_score_name) + '.npy'
    #np.save(os.path.join(result_path, curr_score_name), score)
        
