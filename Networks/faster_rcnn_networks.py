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
from Lib.anchor_target_layer import anchor_target_layer as anchor_target_layer_py

import tensorflow as tf

class rpn:
    '''
    Region Proposal Network (RPN): Takes convolutional feature maps (TensorBase 
    Layers object) from the last layer and proposes bounding boxes for objects.
    '''
    def __init__(self,featureMaps,gt_boxes,im_dims,flags):
        self.featureMaps = featureMaps
        self.gt_boxes = gt_boxes
        self.im_dims = im_dims
        self.flags = flags
        self._network()
        
    def _network(self):
        _num_anchors = len(self.flags['anchor_scales'])*3    
        
        rpn_layers = Layers(self.featureMaps)
        
        with tf.variable_scope('rpn'):
            # Spatial windowing
            rpn_layers.conv2d(filter_size=3,output_channels=512)
            features = rpn_layers.get_output()
            
            # Box-classification layer (objectness)
            self.bbox_cls_layers = Layers(features)
            self.bbox_cls_layers.conv2d(filter_size=1,output_channels=_num_anchors*2,activation_fn=None)        
            
            # Anchor Target Layer
            self.rpn_cls_score = self.bbox_cls_layers.get_output()
            self.rpn_labels,self.rpn_bbox_targets,self.rpn_bbox_inside_weights,self.rpn_bbox_outside_weights = \
                anchor_target_layer(rpn_cls_score=self.rpn_cls_score,gt_boxes=self.gt_boxes,im_dims=self.im_dims,anchor_scales=self.flags['anchor_scales'])       
            
            # Bounding-Box regression layer (bounding box) + anchors
            self.bbox_reg_layers = Layers(features)
            self.bbox_reg_layers.conv2d(filter_size=1,output_channels=_num_anchors*4,activation_fn=None)
                    
#            return bbox_reg_layers, bbox_cls_layers, rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
    
    def get_bbox_reg(self):
        return self.bbox_reg_layers.get_output()
        
    def get_bbox_cls(self):
        return self.bbox_cls_layers.get_output()
    
    def get_rpn_labels(self):
        return self.rpn_labels
        
    def get_rpn_bbox_targets(self):
        return self.bbox_targets
        
    def get_rpn_bbox_inside_weights(self):
        return self.rpn_bbox_inside_weights
        
    def get_rpn_bbox_outside_weights(self):
        return self.rpn_bbox_outside_weights
        
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



# Convert Python into TensorFlow
def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):
    if isinstance(input[0], tuple):
        input[0] = input[0][0]

    with tf.variable_scope(name):

        rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights = \
            tf.py_func(anchor_target_layer_py,[input[0],input[1],input[2],input[3], _feat_stride, anchor_scales],[tf.float32,tf.float32,tf.float32,tf.float32])

        rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels,tf.int32), name = 'rpn_labels')
        rpn_bbox_targets = tf.convert_to_tensor(rpn_bbox_targets, name = 'rpn_bbox_targets')
        rpn_bbox_inside_weights = tf.convert_to_tensor(rpn_bbox_inside_weights , name = 'rpn_bbox_inside_weights')
        rpn_bbox_outside_weights = tf.convert_to_tensor(rpn_bbox_outside_weights , name = 'rpn_bbox_outside_weights')


        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights