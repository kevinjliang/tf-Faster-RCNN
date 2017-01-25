#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 15:05:05 2017

@author: Kevin Liang

Loss functions
"""

import tensorflow as tf


def rpn_cls_loss(rpn_cls_score,rpn_labels):
    '''
    Calculate the Region Proposal Network classifier loss. Measures how well 
    the RPN is able to propose regions by the performance of its "objectness" 
    classifier.
    
    Standard cross-entropy loss on logits
    '''
    # input shape dimensions
    shape = tf.shape(rpn_cls_score)
    
    # Stack all classification scores into 2D matrix
    rpn_cls_score = tf.transpose(rpn_cls_score,[0,3,1,2])
    rpn_cls_score = tf.reshape(rpn_cls_score,[shape[0],2,shape[3]//2*shape[1],shape[2]])
    rpn_cls_score = tf.transpose(rpn_cls_score,[0,2,3,1])
    rpn_cls_score = tf.reshape(rpn_cls_score,[-1,2])
    
    # Stack labels
    rpn_labels = tf.reshape(rpn_labels,[-1])
    
    # Ignore label=-1 (Neither object nor background: IoU between 0.3 and 0.7)
    rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_labels,-1))),[-1,2])
    rpn_labels = tf.reshape(tf.gather(rpn_labels,tf.where(tf.not_equal(rpn_labels,-1))),[-1])
    
    # Cross entropy error
    rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(rpn_cls_score, rpn_labels))
    
    return rpn_cross_entropy
    
    
def rpn_bbox_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_inside_weights, rpn_outside_weights):
    '''
    Calculate the Region Proposal Network bounding box loss. Measures how well 
    the RPN is able to propose regions by the performance of its localization.

    lam/N_reg * sum_i(p_i^* * L_reg(t_i,t_i^*))

    lam: classification vs bbox loss balance parameter     
    N_reg: Number of anchor locations (~2500)
    p_i^*: ground truth label for anchor (loss only for positive anchors)
    L_reg: smoothL1 loss
    t_i: Parameterized prediction of bounding box
    t_i^*: Parameterized ground truth of closest bounding box
    
    TODO: rpn_inside_weights likely deprecated; might consider obliterating
    '''    
    # Constant for weighting bounding box loss with classification loss
    lam = 10
    
    # How far off was the prediction?
    diff = smoothL1(rpn_bbox_pred - rpn_bbox_targets)
    
    # Only count loss for positive anchors
    rpn_bbox_reg = tf.reduce_mean(tf.mul(rpn_outside_weights,diff))
    
    return lam*rpn_bbox_reg    
    
    
def fast_rcnn_cls_loss(fast_rcnn_cls_score, labels):
    '''
    Calculate the fast RCNN classifier loss. Measures how well the fast RCNN is 
    able to classify objects from the RPN.
    
    Standard cross-entropy loss on logits
    '''
    # Cross entropy error
    fast_rcnn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(tf.squeeze(fast_rcnn_cls_score), labels))
    
    return fast_rcnn_cross_entropy
    
    
def fast_rcnn_bbox_loss(fast_rcnn_bbox_pred, bbox_targets, roi_inside_weights, roi_outside_weights):
    '''
    Calculate the fast RCNN bounding box refinement loss. Measures how well 
    the fast RCNN is able to refine localization.

    lam/N_reg * sum_i(p_i^* * L_reg(t_i,t_i^*))

    lam: classification vs bbox loss balance parameter     
    N_reg: Number of anchor locations (~2500)
    p_i^*: ground truth label for anchor (loss only for positive anchors)
    L_reg: smoothL1 loss
    t_i: Parameterized prediction of bounding box
    t_i^*: Parameterized ground truth of closest bounding box
    
    TODO: rpn_inside_weights likely deprecated; might consider obliterating
    '''  
    # Constant for weighting bounding box loss with classification loss
    lam = 1
    
    # How far off was the prediction?
    diff = smoothL1(fast_rcnn_bbox_pred - bbox_targets)
    
    # Only count loss for positive anchors
    roi_bbox_reg = tf.reduce_mean(tf.mul(roi_outside_weights,diff))
    
    return lam*roi_bbox_reg
    
    
def smoothL1(x):
    '''
    Tensorflow implementation of smooth L1 loss defined in Fast RCNN:
        (https://arxiv.org/pdf/1504.08083v2.pdf)
    
                    0.5 x^2         if |x|<1
    smoothL1(x) = {
                    |x| - 0.5       otherwise
    '''
    conditional = tf.less(tf.abs(x),1)
    
    close = 0.5 * x**2
    far = tf.abs(x) - 0.5

    return tf.select(conditional,close,far)