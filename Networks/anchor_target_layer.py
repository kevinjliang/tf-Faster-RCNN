# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 16:11:17 2017

@author: Kevin Liang (modifications)

Anchor Target Layer: Creates all the anchors in the final convolutional feature
map, assigns anchors to ground truth boxes, and applies labels of "objectness"

Adapted from the official Faster R-CNN repo: 
https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/anchor_target_layer.py
"""

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import sys
sys.path.append('../')
sys.path.append('../../')

from Lib.generate_anchors import generate_anchors

import numpy as np

cfg = {
    'allowed_border': 0    # allow boxes to sit over the edge by a small amount
}

def anchor_target_layer(rpn_cls_score, gt_boxes, im_dims, data, _feat_stride = [16,], anchor_scales = [8, 16, 32]):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    
    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap
    """
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]
    
    # Only minibatch of 1 supported
    assert rpn_cls_score.shape[0] == 1, \
        'Only single item batches are supported'    
    
    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]
    
    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)
    
    # anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= -cfg['_allowed_border']) &
        (all_anchors[:, 1] >= -cfg['_allowed_border']) &
        (all_anchors[:, 2] < im_dims[1] + cfg['_allowed_border']) &  # width
        (all_anchors[:, 3] < im_dims[0] + cfg['_allowed_border'])    # height
    )[0]
    
    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    
    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    labels.fill(-1)
    
    