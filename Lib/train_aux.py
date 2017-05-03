#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 14:27:19 2017

@author: Kevin Liang

Helper functions for preprocessing data and training Faster RCNN 
"""

from .faster_rcnn_config import cfg
from .image_functions import read_image, _applyImageFlips, _applyBboxFlips, image_preprocessing

import numpy as np

    
def randomize_training_order(num_training):
    ''' Generated randomized order in which the bags will be read '''
    order = np.random.permutation(num_training)
    return order
    
def create_feed_dict(data_directory, names, tf_inputs, image_index):
    ''' 
    Create the feed_dict for training for a particular image 
    
    data_directory: Directory to the data. Should be organized as follows:
        |--data_directory/
            |--Annotations/
                |--*.txt (Annotation Files: (x1,y1,x2,y2,l))
            |--Images/
                |--*.[png/jpg] (Image files)
            |--Names/
                |--[train/valid/test].txt (List of data)
    names: list of data files (contents of the above [train/valid/test].txt file, returned by read_names)
    tf_inputs: TensorFlow tensor inputs to the computation graph
            [0] x: the image input (rows, cols, channels)
            [1] im_dims: image dimensions of input (height, width)
            [2] gt_boxes: ground truth boxes (and labels) from the annotations file (x1, y1, x2, y2, label)
    image_index: the index of the image the feed_dict is being created for
    '''
    # Data filenames
    image_file = data_directory + 'Images/' + names[image_index] + cfg.IMAGE_FORMAT
    annotation_file = data_directory + 'Annotations/' + names[image_index] + '.txt'

    # Read data
    image = read_image(image_file)
    gt_bbox = np.loadtxt(annotation_file, ndmin=2)
    
    # Image dimensions
    im_dims = np.array(image.shape[:2]).reshape([1, 2])
    
    # Perform data augmentation operations
    flips = [0, 0]
    if cfg.TRAIN.USE_HORZ_FLIPPED:
        # Randomly flip horizontally
        flips[0] = np.random.binomial(1, 0.5)
    if cfg.TRAIN.USE_VERT_FLIPPED:
        # Randomly flip vertically
        flips[1] = np.random.binomial(1, 0.5)

    if cfg.TRAIN.USE_HORZ_FLIPPED or cfg.TRAIN.USE_HORZ_FLIPPED:
        image = _applyImageFlips(image, flips)
        gt_bbox = _applyBboxFlips(gt_bbox, im_dims, flips)

    # Applies dataset-specific pre-processing to image
    image = image_preprocessing(image)

    # Create TensorFlow feed dictionary
    feed_dict = {tf_inputs[0]: image, tf_inputs[1]: im_dims, tf_inputs[2]: gt_bbox}
                 
    return feed_dict
