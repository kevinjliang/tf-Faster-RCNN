# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 16:14:48 2016

@author: Kevin Liang

Faster R-CNN detection and classification networks.

Contains the Region Proposal Network (RPN), ROI proposal layer, and the RCNN.

TODO: -Split off these three networks into their own files OR add to Layers
"""

import sys

sys.path.append('../')

from Lib.faster_rcnn_config import cfg
from Lib.loss_functions import rpn_cls_loss, rpn_bbox_loss, fast_rcnn_cls_loss, fast_rcnn_bbox_loss
from Lib.roi_pool import roi_pool
from Lib.rpn_softmax import rpn_softmax
from Networks.anchor_target_layer import anchor_target_layer
from Networks.proposal_layer import proposal_layer
from Networks.proposal_target_layer import proposal_target_layer

import tensorflow.contrib.layers as tcl
import tensorflow as tf


class rpn:
    '''
    Region Proposal Network (RPN): From the convolutional feature maps of the 
    last layer, generate bounding boxes relative to anchor boxes and give an 
    "objectness" score to each

    In evaluation mode (eval_mode==True), gt_boxes should be None.
    '''

    def __init__(self, featureMaps, gt_boxes, im_dims, _feat_stride, eval_mode):
        self.featureMaps = featureMaps
        self.gt_boxes = gt_boxes
        self.im_dims = im_dims
        self._feat_stride = _feat_stride
        self.anchor_scales = cfg.RPN_ANCHOR_SCALES
        self.eval_mode = eval_mode
        
        self._network()

    def _network(self):
        # There shouldn't be any gt_boxes if in evaluation mode
        if self.eval_mode is True:
            assert self.gt_boxes is None, \
                'Evaluation mode should not have ground truth boxes (or else what are you detecting for?)'

        _num_anchors = len(self.anchor_scales)*3

        features = self.featureMaps
        with tf.variable_scope('rpn'):
            # Spatial windowing
            for i in range(len(cfg.RPN_OUTPUT_CHANNELS)):
                # 2D Conv -> Batch Normalization -> ReLU
                features = tcl.conv2d(inputs=features, 
                                      num_outputs=cfg.RPN_OUTPUT_CHANNELS[i],
                                      kernel_size=cfg.RPN_FILTER_SIZES[i], 
                                      activation_fn=tf.nn.relu,
                                      normalizer_fn=tcl.batch_norm, 
                                      normalizer_params={'is_training':not self.eval_mode})
                
            # Box-classification layer (objectness)
            with tf.variable_scope('cls'):
                # Only 2D Conv
                self.rpn_cls_score = tcl.conv2d(inputs=features, 
                                                num_outputs=_num_anchors*2, 
                                                kernel_size=1,
                                                activation_fn=None)

            # Anchor Target Layer (anchors and deltas)
            with tf.variable_scope('target'):
                # Only calculate targets in train mode. No ground truth boxes in evaluation mode
                if self.eval_mode is False:
                    self.rpn_labels, self.rpn_bbox_targets, self.rpn_bbox_inside_weights, self.rpn_bbox_outside_weights = \
                        anchor_target_layer(rpn_cls_score=self.rpn_cls_score, gt_boxes=self.gt_boxes, im_dims=self.im_dims,
                                            _feat_stride=self._feat_stride, anchor_scales=self.anchor_scales)

            # Bounding-Box regression layer (bounding box predictions)
            with tf.variable_scope('bbox'):
                # Only 2D Conv
                self.rpn_bbox_pred = tcl.conv2d(inputs=features, 
                                                num_outputs=_num_anchors*4, 
                                                kernel_size=1,
                                                activation_fn=None)

    # Get functions
    def get_rpn_cls_score(self):
        return self.rpn_cls_score

    def get_rpn_labels(self):
        assert self.eval_mode is False, 'No RPN labels without ground truth boxes'
        return self.rpn_labels

    def get_rpn_bbox_pred(self):
        return self.rpn_bbox_pred

    def get_rpn_bbox_targets(self):
        assert self.eval_mode is False, 'No RPN bounding box targets without ground truth boxes'
        return self.rpn_bbox_targets

    def get_rpn_bbox_inside_weights(self):
        assert self.eval_mode is False, 'No RPN inside weights without ground truth boxes'
        return self.rpn_bbox_inside_weights

    def get_rpn_bbox_outside_weights(self):
        assert self.eval_mode is False, 'No RPN outside weights without ground truth boxes'
        return self.rpn_bbox_outside_weights

    # Loss functions
    def get_rpn_cls_loss(self):
        assert self.eval_mode is False, 'No RPN cls loss without ground truth boxes'
        rpn_cls_score = self.get_rpn_cls_score()
        rpn_labels = self.get_rpn_labels()
        return rpn_cls_loss(rpn_cls_score, rpn_labels)

    def get_rpn_bbox_loss(self):
        assert self.eval_mode is False, 'No RPN bbox loss without ground truth boxes'
        rpn_bbox_pred = self.get_rpn_bbox_pred()
        rpn_bbox_targets = self.get_rpn_bbox_targets()
        rpn_bbox_inside_weights = self.get_rpn_bbox_inside_weights()
        rpn_bbox_outside_weights = self.get_rpn_bbox_outside_weights()
        return rpn_bbox_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)


class roi_proposal:
    '''
    Propose highest scoring boxes to the RCNN classifier

    In evaluation mode (eval_mode==True), gt_boxes should be None.
    '''

    def __init__(self, rpn_net, gt_boxes, im_dims, eval_mode):
        self.rpn_net = rpn_net
        self.rpn_cls_score = rpn_net.get_rpn_cls_score()
        self.rpn_bbox_pred = rpn_net.get_rpn_bbox_pred()
        self.gt_boxes = gt_boxes
        self.im_dims = im_dims
        self.num_classes = cfg.NUM_CLASSES
        self.anchor_scales = cfg.RPN_ANCHOR_SCALES
        self.eval_mode = eval_mode
        
        self._network()

    def _network(self):
        # There shouldn't be any gt_boxes if in evaluation mode
        if self.eval_mode is True:
            assert self.gt_boxes is None, \
                'Evaluation mode should not have ground truth boxes (or else what are you detecting for?)'
        
        with tf.variable_scope('roi_proposal'):
            # Convert scores to probabilities
            self.rpn_cls_prob = rpn_softmax(self.rpn_cls_score)
    
            # Determine best proposals
            key = 'TRAIN' if self.eval_mode is False else 'TEST'
            self.blobs = proposal_layer(rpn_bbox_cls_prob=self.rpn_cls_prob, rpn_bbox_pred=self.rpn_bbox_pred,
                                        im_dims=self.im_dims, cfg_key=key, _feat_stride=self.rpn_net._feat_stride,
                                        anchor_scales=self.anchor_scales)
    
            if self.eval_mode is False:
                # Calculate targets for proposals
                self.rois, self.labels, self.bbox_targets, self.bbox_inside_weights, self.bbox_outside_weights = \
                    proposal_target_layer(rpn_rois=self.blobs, gt_boxes=self.gt_boxes,
                                          _num_classes=self.num_classes)

    def get_rois(self):
        return self.rois if self.eval_mode is False else self.blobs

    def get_labels(self):
        assert self.eval_mode is False, 'No labels without ground truth boxes'
        return self.labels

    def get_bbox_targets(self):
        assert self.eval_mode is False, 'No bounding box targets without ground truth boxes'
        return self.bbox_targets

    def get_bbox_inside_weights(self):
        assert self.eval_mode is False, 'No RPN inside weights without ground truth boxes'
        return self.bbox_inside_weights

    def get_bbox_outside_weights(self):
        assert self.eval_mode is False, 'No RPN outside weights without ground truth boxes'
        return self.bbox_outside_weights


class fast_rcnn:
    '''
    Crop and resize areas from the feature-extracting CNN's feature maps
    according to the ROIs generated from the ROI proposal layer
    '''

    def __init__(self, featureMaps, roi_proposal_net, eval_mode):
        self.featureMaps = featureMaps
        self.roi_proposal_net = roi_proposal_net
        self.rois = roi_proposal_net.get_rois()
        self.im_dims = roi_proposal_net.im_dims
        self.num_classes = cfg.NUM_CLASSES
        self.eval_mode = eval_mode
        
        self._network()

    def _network(self):
        with tf.variable_scope('fast_rcnn'):

            # ROI pooling
            pooled_features = roi_pool(self.featureMaps, self.rois, self.im_dims)

            # Fully Connect layers (with dropout)
            with tf.variable_scope('fc'):
                features = tcl.flatten(pooled_features)
                for i in range(len(cfg.FRCNN_FC_HIDDEN)):
                    # FC -> Batch Normalization -> ReLU [Commented out:-> Dropout]
                    features = tcl.fully_connected(inputs=features, 
                                                   num_outputs=cfg.FRCNN_FC_HIDDEN[i],
                                                   activation_fn=tf.nn.relu,
                                                   normalizer_fn=tcl.batch_norm, 
                                                   normalizer_params={'is_training':not self.eval_mode})
#                    features = tcl.dropout(inputs=features,
#                                           keep_prob=cfg.FRCNN_DROPOUT_KEEP_RATE,
#                                           is_training=not self.eval_mode)                    

            # Classifier score
            with tf.variable_scope('cls'):
                self.rcnn_cls_score = tcl.fully_connected(inputs=features, 
                                                          num_outputs=self.num_classes,
                                                          activation_fn=None)
            # Bounding Box refinement
            with tf.variable_scope('bbox'):
                self.rcnn_bbox_refine = tcl.fully_connected(inputs=features, 
                                                            num_outputs=self.num_classes*4,
                                                            activation_fn=None)

    # Get functions
    def get_cls_score(self):
        return self.rcnn_cls_score

    def get_cls_prob(self):
        logits = self.get_cls_score()
        return tf.nn.softmax(logits)

    def get_bbox_refinement(self):
        return self.rcnn_bbox_refine

    # Loss functions
    def get_fast_rcnn_cls_loss(self):
        assert self.eval_mode is False, 'No Fast RCNN cls loss without ground truth boxes'
        fast_rcnn_cls_score = self.get_cls_score()
        labels = self.roi_proposal_net.get_labels()
        return fast_rcnn_cls_loss(fast_rcnn_cls_score, labels)

    def get_fast_rcnn_bbox_loss(self):
        assert self.eval_mode is False, 'No Fast RCNN bbox loss without ground truth boxes'
        fast_rcnn_bbox_pred = self.get_bbox_refinement()
        bbox_targets = self.roi_proposal_net.get_bbox_targets()
        roi_inside_weights = self.roi_proposal_net.get_bbox_inside_weights()
        roi_outside_weights = self.roi_proposal_net.get_bbox_outside_weights()
        return fast_rcnn_bbox_loss(fast_rcnn_bbox_pred, bbox_targets, roi_inside_weights, roi_outside_weights)
