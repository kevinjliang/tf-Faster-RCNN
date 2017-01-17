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
    '''
    # input shape dimensions
    shape = tf.shape(rpn_cls_score)
    
    # Stack all classification scores into 2D matrix
    rpn_cls_score = tf.transpose(rpn_cls_score,[0,3,1,2])
    rpn_cls_score = tf.reshape(rpn_cls_score,[shape[0],2,shape[3]/2*shape[1],shape[2]])
    rpn_cls_score = tf.transpose(rpn_cls_score,[0,2,3,1])
    rpn_cls_score = tf.reshape(rpn_cls_score,[-1,2])
    
    # Stack labels
    rpn_labels = tf.reshape(rpn_labels,[-1])
    
    # Ignore label=-1
    rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score,tf.where(tf.not_equal(rpn_labels,-1))),[-1,2])
    rpn_labels = tf.reshape(tf.gather(rpn_labels,tf.where(tf.not_equal(rpn_labels,-1))),[-1])
    
    # Cross entropy error
    rpn_cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(rpn_cls_score, rpn_labels))
    
    return rpn_cross_entropy
    
    
def rpn_bbox_loss(rpn_bbox_pred, rpn_bbox_targets):
    
    # Constant for weighting bounding box loss with classification loss
    lam = 10
    
    
def fast_rcnn_cls_loss(fast_rcnn_cls_score, labels):
    print("TODO")
    
    
def fast_rcnn_bbox_loss(fast_rcnn_bbox_pred, bbox_targets):
    print("TODO")
    
    
    
def smoothL1(x):
    '''
    Smooth L1 loss defined in Fast RCNN (https://arxiv.org/pdf/1504.08083v2.pdf)
    
                    0.5 x^2         if |x|<1
    smoothL1(x) = {
                    |x| - 0.5       otherwise
    '''
    conditional = tf.less(tf.abs(x),1)
    
    close = 0.5 * x**2
    far = tf.abs(x) - 0.5

    return tf.select(conditional,close,far)