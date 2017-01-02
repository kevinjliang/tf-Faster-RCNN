# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 16:14:48 2016

@author: Kevin Liang

Faster R-CNN detection and classification networks.

Contains the Region Proposal Network (RPN), ROI proposal layer, and the RCNN.
"""

import sys
sys.path.append('../')

from Lib.TensorBase.tensorbase.base import Layers
from Lib.anchor_target_layer import anchor_target_layer

import tensorflow as tf

def rpn(featureMaps,gt_boxes,im_dims,flags):
    '''
    Region Proposal Network (RPN): Takes convolutional feature maps (TensorBase 
    Layers object) from the last layer and proposes bounding boxes for objects.
    
    TODO: Convert this into a Layers object, or just into an object.
    '''
    _num_anchors = len(flags['anchor_scales'])*3    
    
    rpn_layers = Layers(featureMaps)
    
    with tf.variable_scope('rpn'):
        # Spatial windowing
        rpn_layers.conv2d(filter_size=3,output_channels=512)
        features = rpn_layers.get_output()
        
        # Box-classification layer (objectness)
        bbox_cls_layers = Layers(features)
        bbox_cls_layers.conv2d(filter_size=1,output_channels=_num_anchors*2,activation_fn=None)        
        
        # Anchor Target Layer
        rpn_cls_score = bbox_cls_layers.get_output()
        rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = \
            anchor_target_layer(rpn_cls_score=rpn_cls_score,gt_boxes=gt_boxes,im_dims=im_dims,anchor_scales=flags['anchor_scales'])       
        
        # Bounding-Box regression layer (bounding box) + anchors
        bbox_reg_layers = Layers(features)
        bbox_reg_layers.conv2d(filter_size=1,output_channels=_num_anchors*4,activation_fn=None)
                
        return bbox_reg_layers, bbox_cls_layers, rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
        
def roi_proposal(bbox,gt_bbox):
    '''
    
    '''
    
    return 
    # return rois
    
    
def rcnn(featureMaps,rois):
    '''
    Crop and resize areas from the feature-extracting CNN's feature maps 
    according to the ROIs generated from the ROI proposal layer
    '''
    
    return
        