# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 16:14:48 2016

@author: Kevin Liang

Faster R-CNN detection and classification networks.

Contains the Region Proposal Network (RPN), ROI proposal layer, and the RCNN.
"""

import sys
sys.path.append('../../')

from TensorBase.tensorbase.base import Layers

import tensorflow as tf

anchor_scales = [1,2,3]
num_anchors = len(anchor_scales)*3

def rpn(featureMaps):
    '''
    Region Proposal Network (RPN): Takes convolutional feature maps (TensorBase 
    Layers object) from the last layer and proposes bounding boxes for objects.
    '''
    rpn_layers = Layers(featureMaps)
    
    with tf.variable_scope('rpn'):
        # Spatial windowing
        rpn_layers.conv2d(filter_size=3,output_channels=512)
        features = rpn_layers.get_output()
        
        # Bounding-Box regression layer (bounding box) + anchors
        bbox_reg_layers = Layers(features)
        bbox_reg_layers.conv2d(filter_size=1,output_channels=num_anchors*2,activation_fn=None)
        
        ###ANCHORS###
        
        # Box-classification layer (objectness)
        bbox_cls_layers = Layers(features)
        bbox_cls_layers.conv2d(filter_size=1,output_channels=num_anchors*4,activation_fn=None)
        
        return bbox_reg_layers, bbox_cls_layers
        
def roi_proposal():
    '''
    
    '''
    
    return
    
    
def rcnn():
    '''
    
    '''
    
    return
        