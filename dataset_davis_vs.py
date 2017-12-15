"""
The DAVIS dataset wrapper for One-Shot Mudulation Network
"""
from PIL import Image
from scipy import ndimage
import os
import numpy as np
import sys
import random
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
from util import get_mask_bbox, get_gb_image, to_bgr, mask_image, adaptive_crop_box, get_dilate_structure, perturb_mask, get_scaled_box
class Dataset:
    def __init__(self, train_list, test_list, args,
            data_aug=False):
        """Initialize the Dataset object
        Args:
        train_list: TXT file or list with the paths of the images to use for training (Images must be between 0 and 255)
        test_list: TXT file or list with the paths of the images to use for testing (Images must be between 0 and 255)
        Returns:
        """
        # Define types of data augmentation
        random.seed(1234)
        self.data_aug = data_aug
        self.data_aug_flip = data_aug
        self.data_aug_scales = args.data_aug_scales
        self.use_original_mask = args.use_original_mask
        self.random_crop_ratio = args.random_crop_ratio
        self.vg_random_rotate_angle = args.vg_random_rotate_angle
        self.vg_random_crop_ratio = args.vg_random_crop_ratio
        self.sg_center_perturb_ratio = args.sg_center_perturb_ratio
        self.sg_std_perturb_ratio = args.sg_std_perturb_ratio
        self.multiclass = args.data_version == 2017
        self.adaptive_crop_testing = args.adaptive_crop_testing
        self.train_list = train_list
        self.test_list = test_list
        self.train_ptr = 0
        self.test_ptr = 0
        self.train_size = len(train_list)
        print '#training samples', self.train_size
        self.test_size = len(test_list)
        self.train_idx = np.arange(self.train_size)
        self.test_idx = np.arange(self.test_size)
        self.crf_infer_steps = 5
        self.dilate_structure = get_dilate_structure(5)
        np.random.shuffle(self.train_idx)
        self.size = args.im_size
        self.crop_size = 300
        self.mean_value = np.array((104, 117, 123))
        self.guide_size = (224, 224)

    def next_batch(self, batch_size, phase):
        """Get next batch of image (path) and labels
        Args:
        batch_size: Size of the batch
        phase: Possible options:'train' or 'test'
        Returns in training:
        images: Numpy arrays of the images
        labels: Numpy arrays of the labels
        Returns in testing:
        images: Numpy array of the images
        path: List of image paths
        """
        if phase == 'train':
            if self.train_ptr + batch_size <= self.train_size:
                idx = np.array(self.train_idx[self.train_ptr:self.train_ptr + batch_size])
                self.train_ptr += batch_size
            else:
                np.random.shuffle(self.train_idx)
                new_ptr = batch_size
                idx = np.array(self.train_idx[:new_ptr])
                self.train_ptr = new_ptr
            guide_images = []
            gb_images = []
            images = []
            labels = []
            if self.data_aug_scales:
                scale = random.choice(self.data_aug_scales)
                new_size = (int(self.size[0] * scale), int(self.size[1] * scale))
            for i in idx:
                sample = self.train_list[i]
                if len(sample) == 4:
                    # guide image is both for appearance and location guidance
                    guide_image = Image.open(sample[0])
                    guide_label = Image.open(sample[1])
                    image = Image.open(sample[2])
                    label = Image.open(sample[3])
                    ref_label = guide_label
                else:
                    # guide image is only for appearance guidance, ref label is only for location guidance
                    guide_image = Image.open(sample[0])
                    guide_label = Image.open(sample[1])
                    #guide_image = Image.open(sample[2])
                    ref_label = Image.open(sample[2])
                    image = Image.open(sample[3])
                    label = Image.open(sample[4])
                if self.data_aug:
                    image, label, ref_label_new = \
                            self.data_augmentation(image, label, ref_label, new_size)
                bbox = get_mask_bbox(np.array(guide_label))
                if not self.use_original_mask:
                    gb_image = get_gb_image(np.array(ref_label_new),center_perturb=self.sg_center_perturb_ratio, 
                            std_perturb=self.sg_std_perturb_ratio)
                else:
                    gb_image = perturb_mask(np.array(ref_label_new))
                    gb_image = ndimage.morphology.binary_dilation(gb_image, 
                            structure=self.dilate_structure) * 255
                guide_image = guide_image.crop(bbox)
                guide_label = guide_label.crop(bbox)
                guide_image = guide_image.resize(self.guide_size, Image.BILINEAR)
                
                guide_label = guide_label.resize(self.guide_size, Image.NEAREST)
                image_data = np.array(image, dtype=np.float32)
                label_data = np.array(label, dtype=np.uint8) > 0 
                guide_image_data = np.array(guide_image, dtype=np.float32)
                guide_image_data = to_bgr(guide_image_data)
                image_data = to_bgr(image_data)
                guide_image_data -= self.mean_value
                image_data -= self.mean_value
                guide_label_data = np.array(guide_label,dtype=np.uint8)
                guide_image_data = mask_image(guide_image_data, guide_label_data)
                guide_images.append(guide_image_data)
                gb_images.append(gb_image)
                images.append(image_data)
                labels.append(label_data)
            images = np.array(images)
            gb_images = np.array(gb_images)[..., np.newaxis]
            labels = np.array(labels)[..., np.newaxis]
            guide_images = np.array(guide_images)
            return guide_images, gb_images, images, labels
        elif phase == 'test':
            guide_images = []
            gb_images = []
            images = []
            image_paths = []
            self.crop_boxes = []
            self.images = []
            if self.test_ptr + batch_size < self.test_size:
                idx = np.array(self.test_idx[self.test_ptr:self.test_ptr + batch_size])
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size) % self.test_size
                idx = np.hstack((self.test_idx[self.test_ptr:], self.test_idx[:new_ptr]))
                self.test_ptr = new_ptr
            for i in idx:
                sample = self.test_list[i]
                if len(sample) == 3:
                    guide_image = Image.open(sample[0])
                    guide_label = Image.open(sample[1])
                    image = Image.open(sample[2])
                    ref_label = guide_label
                    if self.multiclass:
                        ref_name = os.path.join(*(sample[1].split('/')[-3:-1] + [sample[2].split('/')[-1]]))
                    else:
                        ref_name = os.path.join(sample[1].split('/')[-2], sample[2].split('/')[-1])
                else:
                    guide_image = Image.open(sample[0])
                    guide_label = Image.open(sample[1])
                    ref_label = Image.open(sample[2])
                    image = Image.open(sample[3])
                    if self.multiclass:
                        ref_name = os.path.join(*(sample[1].split('/')[-3:-1] + [sample[3].split('/')[-1]]))
                    else:
                        ref_name = os.path.join(sample[1].split('/')[-2], sample[3].split('/')[-1])
                if len(self.size) == 2:
                    self.new_size = self.size
                else:
                    # resize short size of image to self.size[0]
                    resize_ratio = max(float(self.size[0])/image.size[0], float(self.size[0])/image.size[1])
                    self.new_size = (int(resize_ratio * image.size[0]), int(resize_ratio * image.size[1]))
                ref_label = ref_label.resize(self.new_size, Image.NEAREST)
                ref_label_data = np.array(ref_label) / 255
                gb_image = get_gb_image(ref_label_data, center_perturb=0, std_perturb=0)
                image_ref_crf = image.resize(self.new_size, Image.BILINEAR)
                self.images.append(np.array(image_ref_crf))
                if self.adaptive_crop_testing:
                    if np.any(ref_label_data):
                        crop_box = adaptive_crop_box(ref_label_data)
                    else:
                        # use the whole image
                        crop_box = (0,0, ref_label_data.shape[1], ref_label_data.shape[0])
                    crop_w, crop_h = crop_box[2] - crop_box[0], crop_box[3] - crop_box[1]
                    gb_image = gb_image[crop_box[1]: crop_box[3], crop_box[0]: crop_box[2]]
                    
                    scaled_box = get_scaled_box(crop_box, image.size, self.new_size)
                    image = image.crop(scaled_box)
                    # resize short size of image to self.crop_size
                    resize_ratio = max(float(self.crop_size) / crop_w, float(self.crop_size) / crop_h)
                    resize_ratio = max(1, resize_ratio)
                    new_size = (int(resize_ratio * crop_w + 0.5), int(resize_ratio * crop_h + 0.5))
                    self.crop_boxes.append(crop_box)
                    gb_image = ndimage.zoom(gb_image, resize_ratio, mode='nearest')
                    
                    image = image.resize(new_size, Image.BILINEAR)
                else:
                    image = image.resize(self.new_size, Image.BILINEAR)
                if self.use_original_mask:
                    gb_image = ndimage.morphology.binary_dilation(ref_label_data, 
                            structure=self.dilate_structure) * 255
                image_data = np.array(image, dtype=np.float32)

                # process visual guide images
                bbox = get_mask_bbox(np.array(guide_label))
                scaled_box = get_scaled_box(bbox, guide_image.size, guide_label.size)
                guide_image = guide_image.crop(scaled_box)
                guide_label = guide_label.crop(bbox)
                guide_image = guide_image.resize(self.guide_size, Image.BILINEAR)
                guide_label = guide_label.resize(self.guide_size, Image.NEAREST)
                guide_image_data = np.array(guide_image, dtype=np.float32)
                image_data = to_bgr(image_data)
                guide_image_data = to_bgr(guide_image_data)
                guide_image_data -= self.mean_value
                image_data -= self.mean_value
                guide_label_data = np.array(guide_label, dtype=np.uint8)
                guide_image_data = mask_image(guide_image_data, guide_label_data)
                guide_images.append(guide_image_data)
                gb_images.append(gb_image)
                images.append(image_data)
                image_paths.append(ref_name)
            images = np.array(images)
            gb_images = np.array(gb_images)[...,np.newaxis]
            guide_images = np.array(guide_images) 
            return guide_images, gb_images, images, image_paths
        else:
            return None, None, None, None
    
    def data_augmentation(self, im, label, guide_label, new_size):
        im = im.resize(new_size, Image.BILINEAR)
        label = label.resize(new_size, Image.NEAREST)
        guide_label = guide_label.resize(new_size, Image.NEAREST)
        if self.data_aug_flip:
            if random.random() > 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
                guide_label = guide_label.transpose(Image.FLIP_LEFT_RIGHT)
        return im, label, guide_label
    
    def crf_processing(self, image, label, soft_label=False):
        crf = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)
        if not soft_label:
            unary = unary_from_labels(label, 2, gt_prob=0.9, zero_unsure=False)
        else:
            if len(label.shape)==2:
                p_neg = 1.0 - label
                label = np.concatenate((p_neg[...,np.newaxis], label[...,np.newaxis]), axis=2)
            label = label.transpose((2,0,1))
            unary = unary_from_softmax(label)
        crf.setUnaryEnergy(unary)
        crf.addPairwiseGaussian(sxy=(3,3), compat=3)
        crf.addPairwiseBilateral(sxy=(40, 40), srgb=(5, 5, 5), rgbim=image, compat=10)
        crf_out = crf.inference(self.crf_infer_steps)

        # Find out the most probable class for each pixel.
        return np.argmax(crf_out, axis=0).reshape((image.shape[0], image.shape[1]))

    # restore the score map when using adaptive cropping test
    def restore_crop(self, res):
        n_samples = res.shape[0]
        channels = res.shape[3]
        assert(channels == 1)
        restored = np.zeros((n_samples, self.size[1], self.size[0], channels), dtype=np.float32)
        for res_sample, restored_sample, crop_box in zip(res, restored, self.crop_boxes):
            #print 'score range before restore', np.amax(res_sample), np.amin(res_sample)
            res_im = Image.fromarray(res_sample[:,:,0], mode='F')
            res_sample = np.array(res_im.resize((crop_box[2] - crop_box[0], crop_box[3] - crop_box [1]), Image.BILINEAR))
            #print 'score range after restore', np.amax(res_sample), np.amin(res_sample)
            restored_sample[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2],0] = res_sample
        return restored
    
    def get_train_size(self):
        return self.train_size

    def get_test_size(self):
        return self.test_size

    def train_img_size(self):
        return self.size
    
    def reset_idx(self):
        self.train_ptr = 0
        self.test_ptr = 0
