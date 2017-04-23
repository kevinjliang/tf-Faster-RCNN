# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 19:25:41 2017

@author: Kevin Liang (modifications)

Proposal Layer: Applies the Region Proposal Network's (RPN) predicted deltas to
each of the anchors, removes unsuitable boxes, and then ranks them by their
"objectness" scores. Non-maximimum suppression removes proposals of the same 
object, and the top proposals are returned.

Adapted from the official Faster R-CNN repo: 
https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/rpn/proposal_layer.py
"""

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
import tensorflow as tf

from Lib.bbox_transform import bbox_transform_inv, clip_boxes
from Lib.faster_rcnn_config import cfg
from Lib.generate_anchors import generate_anchors
from Lib.nms_wrapper import nms


def proposal_layer(rpn_bbox_cls_prob, rpn_bbox_pred, im_dims, cfg_key, _feat_stride, anchor_scales):
    return tf.reshape(tf.py_func(_proposal_layer_py,[rpn_bbox_cls_prob, rpn_bbox_pred, im_dims[0], cfg_key, _feat_stride, anchor_scales], [tf.float32]),[-1,5])


def _proposal_layer_py(rpn_bbox_cls_prob, rpn_bbox_pred, im_dims, cfg_key, _feat_stride, anchor_scales):
    '''
    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)
    
    '''
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]
    rpn_bbox_cls_prob = np.transpose(rpn_bbox_cls_prob,[0,3,1,2])
    rpn_bbox_pred = np.transpose(rpn_bbox_pred,[0,3,1,2])

    # Only minibatch of 1 supported
    assert rpn_bbox_cls_prob.shape[0] == 1, \
        'Only single item batches are supported' 
    
    if cfg_key == 'TRAIN':
        pre_nms_topN  = cfg.TRAIN.RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg.TRAIN.RPN_POST_NMS_TOP_N
        nms_thresh    = cfg.TRAIN.RPN_NMS_THRESH
        min_size      = cfg.TRAIN.RPN_MIN_SIZE
    else: # cfg_key == 'TEST':        
        pre_nms_topN  = cfg.TEST.RPN_PRE_NMS_TOP_N
        post_nms_topN = cfg.TEST.RPN_POST_NMS_TOP_N
        nms_thresh    = cfg.TEST.RPN_NMS_THRESH
        min_size      = cfg.TEST.RPN_MIN_SIZE

    
    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    scores = rpn_bbox_cls_prob[:, _num_anchors:, :, :]
    bbox_deltas = rpn_bbox_pred
    
    # 1. Generate proposals from bbox deltas and shifted anchors
    height, width = scores.shape[-2:]
    
    # Enumerate all shifts
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
                        
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    anchors = _anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))
    
    # Same story for the scores:
    #
    # scores are (1, A, H, W) format
    # transpose to (1, H, W, A)
    # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))
    
    # Convert anchors into proposals via bbox transformations
    proposals = bbox_transform_inv(anchors, bbox_deltas)
    
    # 2. clip predicted boxes to image
    proposals = clip_boxes(proposals, im_dims)
    
    # 3. remove predicted boxes with either height or width < threshold
    keep = _filter_boxes(proposals, min_size)
    proposals = proposals[keep, :]
    scores = scores[keep]
    
    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]
    
    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    keep = nms(np.hstack((proposals, scores)), nms_thresh)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]
    proposals = proposals[keep, :]
    scores = scores[keep]

    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
    return blob


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep