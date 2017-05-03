#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:02:22 2017

@author: Kevin Liang
"""

from .faster_rcnn_config import cfg

import numpy as np
from scipy.misc import imread

###############################################################################
# Image processing functions
###############################################################################    

def read_image(image_file):

    if cfg.IMAGE_BITDEPTH == 8:
        return imread(image_file)
    else:
        # If not 8-bit, implement your image reader for your data here, 
        # and comment out the NotImplementedError exception 
        raise NotImplementedError

        
def image_preprocessing(image):
    '''
    Applies dataset-specific image pre-processing. Natural image processing 
    (mean subtraction) done by default. Room to add custom preprocessing
    
    Args:
        image (numpy array 2D/3D): image to be processed

    Returns:
        Preprocessed image
    '''

    if cfg.NATURAL_IMAGE:
        image = _rearrange_channels(image)
        image = _subtract_ImageNet_pixel_means(image)
        
    ###########################################################################
    # Optional TODO: Add your own custom preprocessing for your dataset here
    ###########################################################################

    # Expand image to 4 dimensions (batch, height, width, channels)
    if len(image.shape) == 2:
        image = np.expand_dims(np.expand_dims(image, 0), 3)
    else:
        image = np.expand_dims(image, 0)

    return image
    
    
def vis_preprocessing(image):
    '''
    Applies dataset-specific image pre-processing before visualizing outputs.
    
    Default: Do nothing
    
    Args:
        image (numpy array 2D/3D): image to be processed

    Returns:
        Preprocessed image
    '''

    ###########################################################################
    # Optional TODO: Add your own custom preprocessing for your dataset here
    ###########################################################################

    return image
    
    
def _rearrange_channels(image):
    '''
    Flip RGB to BGR for pre-trained weights (OpenCV and Caffe are silly)

    Args:
        image (numpy array 3D)

    Returns:
        Rearranged image
    '''
    return image[:, :, (2, 1, 0)]


def _subtract_ImageNet_pixel_means(image):
    '''
    Subtract ImageNet pixel means found in config file

    Args:
        image (numpy array 3D)

    Returns:
        Demeaned image
    '''
    return image - cfg.PIXEL_MEANS


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
        x1 = im_dims[0,1] - 1 - bbox[:, 2]
        x2 = im_dims[0,1] - 1 - bbox[:, 0]
    if flips[1]:
        y1 = im_dims[0,0] - 1 - bbox[:, 3]
        y2 = im_dims[0,0] - 1 - bbox[:, 1]

    return np.stack((x1, y1, x2, y2, label), axis=1)  
    