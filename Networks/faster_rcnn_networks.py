# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 16:14:48 2016

@author: Kevin Liang

Faster R-CNN detection and classification networks.

Contains the Region Proposal Network (RPN), ROI proposal layer, and the RCNN.

TODO: -Split off these three networks into their own files OR add to Layers
      -Move flags to config file
"""

import sys
sys.path.append('../')

from Lib.TensorBase.tensorbase.base import Layers
from Lib.roi_pool import roi_pool
from Lib.rpn_softmax import rpn_softmax
from Networks.anchor_target_layer import anchor_target_layer 
from Networks.proposal_layer import proposal_layer
from Networks.proposal_target_layer import proposal_target_layer

import tensorflow as tf


class rpn:
    '''
    Region Proposal Network (RPN): From the convolutional feature maps 
    (TensorBase Layers object) of the last layer, generate bounding boxes 
    relative to anchor boxes and give an "objectness" score to each
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

    def get_rpn_cls_score(self):
        return self.rpn_bbox_cls_layers.get_output()
        
    def get_rpn_bbox_pred(self):
        return self.rpn_bbox_pred_layers.get_output()
    
    def get_rpn_labels(self):
        return self.rpn_labels
        
    def get_rpn_bbox_targets(self):
        return self.rpn_bbox_targets
        
    def get_rpn_bbox_inside_weights(self):
        return self.rpn_bbox_inside_weights
        
    def get_rpn_bbox_outside_weights(self):
        return self.rpn_bbox_outside_weights
        
        
class roi_proposal:
    '''
    Propose highest scoring boxes to the rcnn classifier
    '''
    def __init__(self,rpn_cls_score,rpn_bbox_pred,gt_boxes,im_dims,flags):
        self.rpn_cls_score = rpn_cls_score
        self.rpn_bbox_pred = rpn_bbox_pred
        self.gt_boxes = gt_boxes
        self.im_dims = im_dims
        self.flags = flags
        self._network()
        
    def _network(self):
        # Convert scores to probabilities
        self.rpn_cls_prob = rpn_softmax(self.rpn_cls_score)
        
        # Determine best proposals
        self.blobs = proposal_layer(rpn_bbox_cls_prob=self.rpn_cls_prob, rpn_bbox_pred=self.rpn_bbox_pred, im_dims=self.im_dims, cfg_key='TRAIN', _feat_stride=2**5, anchor_scales=self.flags['anchor_scales'])
    
        # Calculate targets for proposals
        self.rois, self.labels, self.bbox_targets, self.bbox_inside_weights, self.bbox_outside_weights = \
            proposal_target_layer(rpn_rois=self.blobs, gt_boxes=self.gt_boxes,_num_classes=self.flags['num_classes'])
    
    def get_rois(self):
        return self.rois
        
    def get_labels(self):
        return self.labels
        
    def get_bbox_targets(self):
        return self.bbox_targets
        
    def get_bbox_inside_weights(self):
        return self.bbox_inside_weights
        
    def get_bbox_outside_weights(self):
        return self.bbox_outside_weights
        
    
class fast_rcnn:
    '''
    Crop and resize areas from the feature-extracting CNN's feature maps 
    according to the ROIs generated from the ROI proposal layer
    '''
    def __init__(self,featureMaps,rois, im_dims, flags):
        self.featureMaps = featureMaps
        self.rois = rois
        self.im_dims = im_dims
        self.flags = flags
        self._network()
            
    def _network(self):
        with tf.variable_scope('fast_rcnn'):
            # ROI pooling
            pooledFeatures = roi_pool(self.featureMaps,self.rois,self.im_dims)
            
            # Fully Connect layers (with dropout)
            with tf.variable_scope('fc'):
                self.rcnn_fc_layers = Layers(pooledFeatures)
                self.rcnn_fc_layers.flatten()
                self.rcnn_fc_layers.fc(output_nodes=4096, keep_prob=0.5)
                self.rcnn_fc_layers.fc(output_nodes=4096, keep_prob=0.5)
                
                hidden = self.rcnn_fc_layers.get_output()
                
            # Classifier score
            with tf.variable_scope('cls'):
                self.rcnn_cls_layers = Layers(hidden)
                self.rcnn_cls_layers.fc(output_nodes=self.flags['num_classes'],activation_fn=None)
    
            # Bounding Box refinement
            with tf.variable_scope('bbox'):
                self.rcnn_bbox_layers = Layers(hidden)
                self.rcnn_bbox_layers.fc(output_nodes=4*self.flags['num_classes'],activation_fn=None)
            
    def get_cls_score(self):
        return self.rcnn_cls_layers.get_output()
        
    def get_bbox_refinement(self):
        return self.rcnn_bbox_layers.get_output()