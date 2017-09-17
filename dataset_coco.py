"""
The dataset class for One-Shot Mudulation Network
"""
from PIL import Image
import os
import numpy as np
import sys
import random
import cPickle
sys.path.append('../coco/PythonAPI')
from pycocotools.coco import COCO
class Dataset:
    def __init__(self, train_anno_file, test_anno_file, train_image_path, test_image_path, data_aug=False, data_aug_scales=[0.8, 1.0, 1.2]):
        """Initialize the Dataset object
        Args:
        train_anno_file: json file for training data
        test_anno_file: json file for testing data
        database_root: Path to the root of the Database
        store_memory: True stores all the training images, False loads at runtime the images
        Returns:
        """
        # Define types of data augmentation
        self.data_aug = data_aug
        self.data_aug_flip = data_aug
        self.data_aug_scales = data_aug_scales
        self.fg_thresh = 0.03
        random.seed(1234)
        self.train_image_path = train_image_path
        self.test_image_path = test_image_path
        
        self.train_data = COCO(train_anno_file)
        self.test_data = COCO(test_anno_file)
        if os.path.exists('cache/train_annos.pkl'):
            self.train_annos = cPickle.load(open('cache/train_annos.pkl', 'rb'))
            self.test_annos = cPickle.load(open('cache/test_annos.pkl', 'rb'))
        else:
            # prefiltering of segmentation instances
            self.train_annos = self.prefilter(self.train_data)
            self.test_annos = self.prefilter(self.test_data)
            cPickle.dump(self.train_annos, open('cache/train_annos.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
            cPickle.dump(self.test_annos, open('cache/test_annos.pkl', 'wb'), cPickle.HIGHEST_PROTOCOL)
        # Init parameters

        self.train_ptr = 0
        self.test_ptr = 0
        self.train_size = len(self.train_annos)
        self.test_size = len(self.test_annos)
        self.train_idx = np.arange(self.train_size) 
        self.test_idx = np.arange(self.test_size)
        self.size = (400,400)
        self.mean_value = np.array((104, 117, 123))
        self.guide_size = (224,224)
        np.random.shuffle(self.train_idx)
    
    def prefilter(self, dataset):
        res_annos = []
        annos = dataset.dataset['annotations']
        for anno in annos:
            # throw away all crowd annotations
            if anno['iscrowd']: continue
            m = dataset.annToMask(anno)
            mask_area = np.count_nonzero(m)
            if mask_area / float(m.shape[0] * m.shape[1]) > self.fg_thresh:
                anno['bbox'] = self.get_mask_bbox(m)
                res_annos.append(anno)
        return res_annos
    
    def get_mask_bbox(self, m, border_pixels=8):
        rows = np.any(m, axis=1)
        cols = np.any(m, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        h,w = m.shape
        ymin = max(0, ymin - border_pixels)
        ymax = min(h-1, ymax + border_pixels)
        xmin = max(0, xmin - border_pixels)
        xmax = min(w-1, xmax + border_pixels)
        return (xmin, ymin, xmax, ymax)

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
            images = []
            labels = []
            guide_images = []
            
            if self.data_aug_scales:
                scale = random.choice(self.data_aug_scales)
                new_size = (int(self.size[0] * scale), int(self.size[1] * scale))
            for i in idx:
                anno = self.train_annos[i]
                image_path = self.train_image_path.format(anno['image_id'])
                image = Image.open(image_path)
                label_data = self.train_data.annToMask(anno).astype(np.uint8)
                label = Image.fromarray(label_data)
                
                guide_image = image.crop(anno['bbox'])
                guide_label = label.crop(anno['bbox'])
                guide_image = guide_image.resize(self.guide_size, Image.BILINEAR)
                guide_label = guide_label.resize(self.guide_size, Image.NEAREST)

                if self.data_aug:
                    image, label = self.data_augmentation(image, label, new_size)
                image_data = np.array(image, dtype=np.float32)
                label_data = np.array(label, dtype=np.float32)
                guide_image_data = np.array(guide_image, dtype=np.float32)
                if len(image_data.shape) < 3:
                    image_data = np.repeat(image_data[...,np.newaxis], 3, axis=2)
                    guide_image_data = np.repeat(guide_image_data[...,np.newaxis], 3, axis=2)
                image_data = image_data[:,:,2::-1] - self.mean_value
                guide_image_data = guide_image_data[:,:,2::-1] - self.mean_value
                guide_label_data = np.array(guide_label, dtype=np.uint8)
                # masking
                for ch in range(guide_image_data.shape[2]):
                    guide_image_data[guide_label_data == 0, ch] = 0
                images.append(image_data)
                labels.append(label_data)
                guide_images.append(guide_image_data)
            images= np.array(images)
            labels = np.array(labels)
            labels = labels[...,np.newaxis]
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
                idx = np.array(self.test_idx[self.test_ptr:] + self.test_idx[:new_ptr])
                self.test_ptr = new_ptr
            for i in idx:
                anno = self.test_annos[i]
                image_path = self.test_image_path.format(anno['image_id'])
                image = Image.open(image_path)
                label_data = self.test_data.annToMask(anno).astype(np.uint8)
                label = Image.fromarray(label_data)
                
                guide_image = image.crop(anno['bbox'])
                guide_label = label.crop(anno['bbox'])
                guide_image = guide_image.resize(self.guide_size, Image.BILINEAR)
                guide_label = guide_label.resize(self.guide_size, Image.NEAREST)
                image = image.resize(self.size, Image.BILINEAR)
                label = label.resize(self.size, Image.NEAREST)
                image_data = np.array(image, dtype=np.float32)
                guide_image_data = np.array(guide_image, dtype=np.float32)
                if len(image_data.shape) < 3:
                    image_data = np.repeat(image_data[...,np.newaxis], 3, axis=2)
                    guide_image_data = np.repeat(guide_image_data[...,np.newaxis], 3, axis=2)
                image_data = image_data[:,:,2::-1] - self.mean_value
                guide_image_data = guide_image_data[:,:,2::-1] - self.mean_value
                guide_label_data = np.array(guide_label, dtype=np.uint8)
                # masking
                for ch in range(guide_image_data.shape[2]):
                    guide_image_data[guide_label_data == 0, ch] = 0
                images.append(image_data)
                image_paths.append(image_path)
                guide_images.append(guide_image_data)
            images= np.array(images)
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
