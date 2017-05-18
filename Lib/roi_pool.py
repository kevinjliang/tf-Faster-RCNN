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

def roi_pool(featureMaps,rois,im_dims):    
    '''
    Regions of Interest (ROIs) from the Region Proposal Network (RPN) are 
    formatted as:
    (image_id, x1, y1, x2, y2)
    
    Note: Since mini-batches are sampled from a single image, image_id = 0s
    '''
    with tf.variable_scope('roi_pool'):
        # Image that the ROI is taken from (minibatch of 1 means these will all be 0)
        box_ind = tf.cast(rois[:,0],dtype=tf.int32)
        
        # ROI box coordinates. Must be normalized and ordered to [y1, x1, y2, x2]
        boxes = rois[:,1:]
        normalization = tf.cast(tf.stack([im_dims[:,1],im_dims[:,0],im_dims[:,1],im_dims[:,0]],axis=1),dtype=tf.float32)
        boxes = tf.div(boxes,normalization)
        boxes = tf.stack([boxes[:,1],boxes[:,0],boxes[:,3],boxes[:,2]],axis=1)  # y1, x1, y2, x2
        
        # ROI pool output size
        crop_size = tf.constant([14,14])
        
        # ROI pool
        pooledFeatures = tf.image.crop_and_resize(image=featureMaps, boxes=boxes, box_ind=box_ind, crop_size=crop_size)
        
        # Max pool to (7x7)
        pooledFeatures = tf.nn.max_pool(pooledFeatures, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return pooledFeatures