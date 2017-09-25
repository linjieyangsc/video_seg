"""
The DAVIS dataset wrapper for One-Shot Mudulation Network
"""
from PIL import Image
import os
import numpy as np
import sys
import random
from util import get_mask_bbox, to_bgr, mask_image
class Dataset:
    def __init__(self, train_list, test_list, data_aug=False, data_aug_scales=[0.5, 0.8, 1]):
        """Initialize the Dataset object
        Args:
        train_list: TXT file or list with the paths of the images to use for training (Images must be between 0 and 255)
        test_list: TXT file or list with the paths of the images to use for testing (Images must be between 0 and 255)
        Returns:
        """
        # Define types of data augmentation
        self.data_aug = data_aug
        self.data_aug_flip = data_aug
        self.data_aug_scales = data_aug_scales
        random.seed(1234)
        
        # Init parameters
        self.train_list = train_list
        self.test_list = test_list
        self.train_ptr = 0
        self.test_ptr = 0
        self.train_size = len(train_list)
        self.test_size = len(test_list)
        self.train_idx = np.arange(self.train_size)
        self.test_idx = np.arange(self.test_size)
        np.random.shuffle(self.train_idx)
        self.size = (854, 480)
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
            if self.train_ptr + batch_size < self.train_size:
                idx = np.array(self.train_idx[self.train_ptr:self.train_ptr + batch_size])
                self.train_ptr += batch_size
            else:
                np.random.shuffle(self.train_idx)
                new_ptr = batch_size
                idx = np.array(self.train_idx[:new_ptr])
                self.train_ptr = new_ptr
            guide_images = []
            images = []
            labels = []
            if self.data_aug_scales:
                scale = random.choice(self.data_aug_scales)
                new_size = (int(self.size[0] * scale), int(self.size[1] * scale))
            for i in idx:
                sample = self.train_list[i]
                guide_image = Image.open(sample[0])
                guide_label = Image.open(sample[1])
                image = Image.open(sample[2])
                label = Image.open(sample[3])
                if self.data_aug:
                    image, label = self.data_augmentation(image, label, new_size)
                bbox = get_mask_bbox(np.array(guide_label))
                guide_image = guide_image.crop(bbox)
                guide_label = guide_label.crop(bbox)
                guide_image = guide_image.resize(self.guide_size, Image.BILINEAR)
                guide_label = guide_label.resize(self.guide_size, Image.NEAREST)
                image_data = np.array(image, dtype=np.float32)
                label_data = np.array(label, dtype=np.uint8) > 0 
                #label_im = Image.fromarray((label_data).astype(np.uint8))
                #print 'save test image'
                #label_im.save('test/'+sample[3].split('/')[-1])
                guide_image_data = np.array(guide_image, dtype=np.float32)
                guide_image_data = to_bgr(guide_image_data)
                image_data = to_bgr(image_data)
                guide_image_data -= self.mean_value
                image_data -= self.mean_value
                guide_label_data = np.array(guide_label,dtype=np.uint8)
                guide_image_data = mask_image(guide_image_data, guide_label_data)
                guide_images.append(guide_image_data)
                images.append(image_data)
                labels.append(label_data)
            images = np.array(images)
            labels = np.array(labels)[..., np.newaxis]
            guide_images = np.array(guide_images)
            return guide_images, images, labels
        elif phase == 'test':
            guide_images = []
            images = []
            image_paths = []
            if self.test_ptr + batch_size < self.test_size:
                idx = np.array(self.test_idx[self.test_ptr:self.test_ptr + batch_size])
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size) % self.test_size
                if new_ptr == 0:
                    idx = np.array(self.test_idx[self.test_ptr:])
                else:
                    idx = np.array(self.test_idx[self.test_ptr:] + self.test_idx[:new_ptr])
                self.test_ptr = new_ptr
            for i in idx:
                sample = self.test_list[i]
                guide_image = Image.open(sample[0])
                guide_label = Image.open(sample[1])
                image = Image.open(sample[2])
                bbox = get_mask_bbox(np.array(guide_label))
                guide_image = guide_image.crop(bbox)
                guide_label = guide_label.crop(bbox)
                guide_image = guide_image.resize(self.guide_size, Image.BILINEAR)
                guide_label = guide_label.resize(self.guide_size, Image.NEAREST)
                image = image.resize(self.size, Image.BILINEAR)
                image_data = np.array(image, dtype=np.float32)
                guide_image_data = np.array(guide_image, dtype=np.float32)
                image_data = to_bgr(image_data)
                guide_image_data = to_bgr(guide_image_data)
                guide_image_data -= self.mean_value
                image_data -= self.mean_value
                guide_label_data = np.array(guide_label, dtype=np.uint8)
                guide_image_data = mask_image(guide_image_data, guide_label_data)
                guide_images.append(guide_image_data)
                images.append(image_data)
                image_paths.append(os.path.join(*(sample[1].split('/')[-3:-1] + [sample[2].split('/')[-1]])))
            images = np.array(images)
            guide_images = np.array(guide_images) 
            return guide_images, images, image_paths
        else:
            return None, None, None
    
    def data_augmentation(self, im, label, new_size):
        im = im.resize(new_size, Image.BILINEAR)
        label = label.resize(new_size, Image.NEAREST)
        if self.data_aug_flip:
            if random.random() > 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return im, label

    def get_train_size(self):
        return self.train_size

    def get_test_size(self):
        return self.test_size

    def train_img_size(self):
        return self.size
