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

from Lib.TensorBase.tensorbase.base import Layers

from Lib.faster_rcnn_config import cfg
from Lib.loss_functions import rpn_cls_loss, rpn_bbox_loss, fast_rcnn_cls_loss, fast_rcnn_bbox_loss
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

        rpn_layers = Layers(self.featureMaps)

        with tf.variable_scope('rpn'):
            # Spatial windowing
            for i in range(len(cfg.RPN_OUTPUT_CHANNELS)):
                rpn_layers.conv2d(filter_size=cfg.RPN_FILTER_SIZES[i], output_channels=cfg.RPN_OUTPUT_CHANNELS[i])
                
            features = rpn_layers.get_output()

            with tf.variable_scope('cls'):
                # Box-classification layer (objectness)
                self.rpn_bbox_cls_layers = Layers(features)
                self.rpn_bbox_cls_layers.conv2d(filter_size=1, output_channels=_num_anchors*2, activation_fn=None)

            with tf.variable_scope('target'):
                # Only calculate targets in train mode. No ground truth boxes in evaluation mode
                if self.eval_mode is False:
                    # Anchor Target Layer (anchors and deltas)
                    rpn_cls_score = self.rpn_bbox_cls_layers.get_output()
                    self.rpn_labels, self.rpn_bbox_targets, self.rpn_bbox_inside_weights, self.rpn_bbox_outside_weights = \
                        anchor_target_layer(rpn_cls_score=rpn_cls_score, gt_boxes=self.gt_boxes, im_dims=self.im_dims,
                                            _feat_stride=self._feat_stride, anchor_scales=self.anchor_scales)

            with tf.variable_scope('bbox'):
                # Bounding-Box regression layer (bounding box predictions)
                self.rpn_bbox_pred_layers = Layers(features)
                self.rpn_bbox_pred_layers.conv2d(filter_size=1, output_channels=_num_anchors*4, activation_fn=None)

    # Get functions
    def get_rpn_cls_score(self):
        return self.rpn_bbox_cls_layers.get_output()

    def get_rpn_labels(self):
        assert self.eval_mode is False, 'No RPN labels without ground truth boxes'
        return self.rpn_labels

    def get_rpn_bbox_pred(self):
        return self.rpn_bbox_pred_layers.get_output()

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
            # No dropout in evaluation mode
            keep_prob = cfg.FRCNN_DROPOUT_KEEP_RATE if self.eval_mode is False else 1.0

            # ROI pooling
            pooledFeatures = roi_pool(self.featureMaps, self.rois, self.im_dims)

            # Fully Connect layers (with dropout)
            with tf.variable_scope('fc'):
                self.rcnn_fc_layers = Layers(pooledFeatures)
                self.rcnn_fc_layers.flatten()
                for i in range(len(cfg.FRCNN_FC_HIDDEN)):
                    self.rcnn_fc_layers.fc(output_nodes=cfg.FRCNN_FC_HIDDEN[i], keep_prob=keep_prob)

                hidden = self.rcnn_fc_layers.get_output()

            # Classifier score
            with tf.variable_scope('cls'):
                self.rcnn_cls_layers = Layers(hidden)
                self.rcnn_cls_layers.fc(output_nodes=self.num_classes, activation_fn=None)

            # Bounding Box refinement
            with tf.variable_scope('bbox'):
                self.rcnn_bbox_layers = Layers(hidden)
                self.rcnn_bbox_layers.fc(output_nodes=self.num_classes*4, activation_fn=None)

    # Get functions
    def get_cls_score(self):
        return self.rcnn_cls_layers.get_output()

    def get_cls_prob(self):
        logits = self.get_cls_score()
        return tf.nn.softmax(logits)

    def get_bbox_refinement(self):
        return self.rcnn_bbox_layers.get_output()

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
