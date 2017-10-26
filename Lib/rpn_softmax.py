# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 02:54:21 2017

@author: Kevin Liang
"""

import tensorflow as tf

def rpn_softmax(rpn_cls_score):
    '''
    Reshape the rpn_cls_score (n,W,H,2k) to take a softmax. Converts scores to 
    probabilities
    
    ex. 9 anchors, n samples minibatch, convolutional feature maps of dims WxH
    
    rpn_cls_score:  (n,W,H,18)
    <transpose>     (n,18,W,H)
    <reshape>       (n,2,9W,H)
    <transpose>     (n,9W,H,2)
    <softmax>       (n,9W,H,2)
    <transpose>     (n,2,9W,H)
    <reshape>       (n,18,W,H)
    <transpose>     (n,W,H,18)
    
    return rpn_cls_prob
    
    TODO: Can probably just take the softmax of a specific index and get rid of
    two tranpsoses
    '''
    with tf.variable_scope('rpn_softmax'):
        # input shape dimensions
        shape = tf.shape(rpn_cls_score)
        
        # Reshape rpn_cls_score to prepare for softmax
        rpn_cls_score = tf.transpose(rpn_cls_score,[0,3,1,2])
        rpn_cls_score = tf.reshape(rpn_cls_score,[shape[0],2,shape[3]//2*shape[1],shape[2]])
        rpn_cls_score = tf.transpose(rpn_cls_score,[0,2,3,1])
        
        # Softmax
        rpn_cls_prob = tf.nn.softmax(rpn_cls_score)
        
        # Reshape back to the original
        rpn_cls_prob = tf.transpose(rpn_cls_prob,[0,3,1,2])
        rpn_cls_prob = tf.reshape(rpn_cls_prob,[shape[0],shape[3],shape[1],shape[2]])
        rpn_cls_prob = tf.transpose(rpn_cls_prob,[0,2,3,1])

    return rpn_cls_prob
        