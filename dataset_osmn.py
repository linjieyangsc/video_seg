"""
The dataset class for One-Shot Mudulation Network
"""
from PIL import Image
import os
import numpy as np
import sys
import random

class Dataset:
    def __init__(self, train_list, test_list, data_aug=False, data_aug_scales=[0.5, 0.8, 1]):
        """Initialize the Dataset object
        Args:
        train_list: TXT file or list with the paths of the images to use for training (Images must be between 0 and 255)
        test_list: TXT file or list with the paths of the images to use for testing (Images must be between 0 and 255)
        database_root: Path to the root of the Database
        store_memory: True stores all the training images, False loads at runtime the images
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

    def next_batch(self, batch_size, phase):
        """Get next batch of image (path) and labels
        Args:
        batch_size: Size of the batch
        phase: Possible options:'train' or 'test'
        Returns in training:
        images: List of images paths if store_memory=False, List of Numpy arrays of the images if store_memory=True
        labels: List of labels paths if store_memory=False, List of Numpy arrays of the labels if store_memory=True
        Returns in testing:
        images: None if store_memory=False, Numpy array of the image if store_memory=True
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
            guide_labels = []
            images = []
            labels = []
            for i in idx:
                if self.data_aug_scales:
                    scale_guide = random.choice(self.data_aug_scales)
                    scale = random.choice(self.data_aug_scales)
                sample = self.train_list[i]
                guide_image = Image.open(sample[0])
                guide_label = Image.open(sample[1])
                image = Image.open(sample[2])
                label = Image.open(sample[3])
                if self.data_aug:
                    guide_image, guide_label = self.data_augmentation(guide_image, guide_label, scale_guide)
                    image, label = self.data_augmentation(image, label, scale)
                guide_images.append(np.array(guide_image))
                guide_labels.append(np.array(guide_label.split()[0]))
                images.append(np.array(image))
                labels.append(np.array(label.split()[0]))
            return guide_images, guide_labels, images, labels
        elif phase == 'test':
            guide_images = []
            guide_labels = []
            images = []
            image_paths = []
            if self.test_ptr + batch_size < self.test_size:
                idx = np.array(self.test_idx[self.test_ptr:self.test_ptr + batch_size])
                self.test_ptr += batch_size
            else:
                new_ptr = (self.test_ptr + batch_size) % self.test_size
                idx = np.array(self.test_idx[self.test_ptr:] + self.test_idx[:new_ptr])
                self.test_ptr = new_ptr
            for i in idx:
                sample = self.test_list[i]
                guide_image = Image.open(sample[0])
                guide_label = Image.open(sample[1])
                image = Image.open(sample[2])
                guide_images.append(np.array(guide_image))
                guide_labels.append(np.array(guide_label.split()[0]))
                images.append(np.array(image))
                image_paths.append(os.path.join(*(sample[1].split('/')[:-1] + [sample[2].split('/')[-1]])))

            return guide_images, guide_labels, images, image_paths
        else:
            return None, None, None, None
    
    def data_augmentation(self, im, label, scale):
        new_size = (int(im.size[0] * scale), int(im.size[1] * scale))
        im = im.resize(new_size)
        label = im.resize(new_size)
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
        width, height = Image.open(self.images_train[self.train_ptr]).size
        return height, width
