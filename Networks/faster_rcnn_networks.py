# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 16:14:48 2016

@author: Kevin Liang

Faster R-CNN detection and classification networks.

Contains the Region Proposal Network (RPN), ROI proposal layer, and the RCNN.

TODO: -Split off these three networks into their own files
      -Move the TensorFlow-ifying code to the layer file
"""

import sys
sys.path.append('../')

from Lib.TensorBase.tensorbase.base import Layers
from Lib.anchor_target_layer import anchor_target_layer 
from Lib.proposal_layer import proposal_layer


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
            self.rpn_bbox_cls_layers = Layers(features)
            self.rpn_bbox_cls_layers.conv2d(filter_size=1,output_channels=_num_anchors*2,activation_fn=None)        
            
            # Anchor Target Layer (anchors and deltas)
            self.rpn_cls_score = self.rpn_bbox_cls_layers.get_output()
            self.rpn_labels,self.rpn_bbox_targets,self.rpn_bbox_inside_weights,self.rpn_bbox_outside_weights = \
                anchor_target_layer(rpn_cls_score=self.rpn_cls_score,gt_boxes=self.gt_boxes,im_dims=self.im_dims,anchor_scales=self.flags['anchor_scales'])       
            
            # Bounding-Box regression layer (bounding box predictions)
            self.rpn_bbox_pred_layers = Layers(features)
            self.rpn_bbox_pred_layers.conv2d(filter_size=1,output_channels=_num_anchors*4,activation_fn=None)

    def get_rpn_bbox_cls(self):
        return self.rpn_bbox_cls_layers.get_output()
        
    def get_rpn_bbox_pred(self):
        return self.rpn_bbox_pred_layers.get_output()
    
    def get_rpn_labels(self):
        return self.rpn_labels
        
    def get_rpn_bbox_targets(self):
        return self.bbox_targets
        
    def get_rpn_bbox_inside_weights(self):
        return self.rpn_bbox_inside_weights
        
    def get_rpn_bbox_outside_weights(self):
        return self.rpn_bbox_outside_weights
        
        
class roi_proposal:
    '''
    
    '''
    def __init__(self,rpn_bbox_cls,gt_boxes,im_dims,flags):
        self.rpn_bbox_cls = rpn_bbox_cls
        self.gt_boxes = gt_boxes
        self.im_dims = im_dims
        self.flags = flags
        self._network()
        
    def _network(self):
        
    
    
    
class rcnn:
    '''
    Crop and resize areas from the feature-extracting CNN's feature maps 
    according to the ROIs generated from the ROI proposal layer
    '''
    def __init__(self,featureMaps,rois):
        self.featureMaps = featureMaps
        self.rois = rois
        self._network()
            
    def _network(self):
        print("TODO")
