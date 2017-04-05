#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 14:27:19 2017

@author: Kevin Liang

Helper functions for preprocessing data and training Faster RCNN 
"""

from .fast_rcnn_config import cfg

import numpy as np
from scipy.misc import imread

    
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
                |--*.png (Image files)
            |--Names/
                |--[train/valid/test].txt (List of data)
    names: list of data files (contents of the above [train/valid/test].txt file, returned by read_names)
    tf_inputs: TensorFlow tensor inputs to the computation graph
            [0] x: the image input (rows, cols, channels)
            [1] im_dims: image dimensions of input (height, width)
            [2] gt_boxes: ground truth boxes (and labels) from the annotations file (x1, y1, x2, y2, label)
    image_index: the index of the image the feed_dict is being crated for
    '''
    # Data filenames
    image_file = data_directory + 'Images/' + names[image_index] + '.png'
    annotation_file = data_directory + 'Annotations/' + names[image_index] + '.txt'

    # Read data
    image = imread(image_file)    
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
        
    # Expand image to 4 dimensions (batch, height, width, channels)
    if len(image.shape) == 2:
        image = np.expand_dims(np.expand_dims(image, 0), 3)
    else:
        image = np.expand_dims(image, 0)
    
    # Create TensorFlow feed dictionary
    feed_dict = {tf_inputs[0]: image, tf_inputs[1]: im_dims, tf_inputs[2]: gt_bbox}
                 
    return feed_dict


    
###############################################################################
# Image processing functions
###############################################################################    
    
def _applyImageFlips(image, flips):
    '''
    Apply left-right and up-down flips to an image
    
    Args:
        image (numpy array 2D/3D): image to be flipped
        flips (tuple):
            [0]: Boolean to flip horizontally
            [1]: Boolean to flip vertically

    Returns:
        Flipped image
    '''
    image = np.fliplr(image) if flips[0] else image
    image = np.flipud(image) if flips[1] else image

    return image
    
    
def _applyBboxFlips(bbox, im_dims, flips):
    '''
    Apply left-right and up-down flips to bounding box
    
    Args:
        bbox (np.2darray): [[x1, y1, x2, y2, label]]
        im_dims (list): (height, width)
        flips (tuple):
            [0]: Boolean to flip horizontally
            [1]: Boolean to flip vertically
            
    Returns:
        Bounding box information for flipped image
    '''
    x1 = bbox[:, 0]    
    y1 = bbox[:, 1]    
    x2 = bbox[:, 2]    
    y2 = bbox[:, 3]    
    label = bbox[:, 4]

    if flips[0]:
        x1 = im_dims[0,1] - bbox[:, 2]
        x2 = im_dims[0,1] - bbox[:, 0]
    if flips[1]:
        y1 = im_dims[0,0] - bbox[:, 3]
        y2 = im_dims[0,0] - bbox[:, 1]

    return np.stack((x1, y1, x2, y2, label), axis=1)  
    