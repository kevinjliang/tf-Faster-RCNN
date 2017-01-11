# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 14:29:57 2017

@author: Kevin Liang

ROI pooling layer: Using tensorflow's crop_and_resize function as replacement.
crop_and_resize expects box indices in normalized coordinates.

Convolutional feature maps are cropped to a constant size of (14,14) and then
maxpooled to (7x7)
"""

import tensorflow as tf
from tensorflow.image import crop_and_resize
from tensorflow.nn import max_pool

def roi_pool(featureMaps,rois,im_dims):    
    '''
    Regions of Interest (ROIs) from the Region Proposal Network (RPN) are 
    formatted as:
    (image_id, x1, y1, x2, y2)
    
    Note: Since mini-batches are sampled from a single image, image_id = 0s
    '''
    # Image that the ROI is taken from (these will all be 0)
    box_ind = rois[:,0]
    
    # ROI box coordinates. Must be normalized and ordered to [y1, x1, y2, x2]
    boxes = rois[:,1:]
    boxes[:,1] = tf.div(boxes[:,1],tf.cast(im_dims[1]),tf.float32)  # normalize x1
    boxes[:,2] = tf.div(boxes[:,2],tf.cast(im_dims[0]),tf.float32)  # normalize y1
    boxes[:,3] = tf.div(boxes[:,3],tf.cast(im_dims[1]),tf.float32)  # normalize x2
    boxes[:,4] = tf.div(boxes[:,4],tf.cast(im_dims[0]),tf.float32)  # normalize y2
    boxes = tf.transpose(boxes,[1,0,3,2])
    
    # ROI pool output size
    crop_size = tf.constant([14,14])
    
    # ROI pool
    pooledFeatures = crop_and_resize(image=featureMaps, boxes=boxes, box_ind=box_ind, crop_size=crop_size)
    
    # Max pool to (7x7)
    pooledFeatures = max_pool(pooledFeatures, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return pooledFeatures