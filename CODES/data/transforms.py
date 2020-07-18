#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Developed by Shangchen Zhou <shangchenzhou@gmail.com>
'''ref: http://pytorch.org/docs/master/torchvision/transforms.html'''

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
import random

class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> transforms.Compose([
    >>>     transforms.CenterCrop(10),
    >>>     transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, tensor_list):
        for t in self.transforms:
            tensor_list = t(tensor_list)
        return tensor_list

class CenterCrop(object):

    def __init__(self, crop_size):
        """Set the height and weight before and after cropping"""
        self.crop_size_h  = crop_size[0]
        self.crop_size_w  = crop_size[1]

    def __call__(self, tensor_list):
        input_size_c, input_size_h, input_size_w = tensor_list[0].shape
        x_start = int(round((input_size_w - self.crop_size_w) / 2.))
        y_start = int(round((input_size_h - self.crop_size_h) / 2.))
        for i in range(len(tensor_list)):
            tensor_list[i] = tensor_list[i][:, y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w]
        return tensor_list

class RandomCrop(object):

    def __init__(self, crop_size):
        """Set the height and weight before and after cropping"""
        self.crop_size_h  = crop_size[0]
        self.crop_size_w  = crop_size[1]

    def __call__(self, tensor_list):
        input_size_c, input_size_h, input_size_w = tensor_list[0].shape
        x_start = random.randint(0, input_size_w - self.crop_size_w)
        y_start = random.randint(0, input_size_h - self.crop_size_h)

        for i in range(len(tensor_list)):
            tensor_list[i] = tensor_list[i][:, y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w]
        return tensor_list

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5 left-right"""

    def __call__(self, tensor_list):
        if random.random() < 0.5:
            '''Change the order of 0 and 1, for keeping the net search direction'''
            for i in range(len(tensor_list)):
                tensor_list[i] = np.copy(np.fliplr(tensor_list[i]))
        return tensor_list

class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5  up-down"""

    def __call__(self, tensor_list):
        if random.random() < 0.5:
            for i in range(len(tensor_list)):
                tensor_list[i] = np.copy(np.flipud(tensor_list[i]))
        return tensor_list

class ColorJitter(object):
    def __init__(self, brightness):
        """brightness [max(0, 1 - brightness), 1 + brightness] or the given [min, max]"""
        """contrast [max(0, 1 - contrast), 1 + contrast] or the given [min, max]"""
        self.brightness = brightness

    def __call__(self, tensor_list):
        # tensor_list[0] = blurred
        if self.brightness > 0:
            brightness_factor = np.random.uniform(-self.brightness, self.brightness)
            mean = np.mean(tensor_list[0])
            tensor_list[0] = tensor_list[0] + mean * brightness_factor

        tensor_list[0] = tensor_list[0].clip(0, 1.0)
        return tensor_list