#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 10:17:41 2017

@author: Kevin Liang (Modifications)

Functions for testing Faster RCNN net after it's been trained
"""

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import sys
sys.path.append('../')

from Lib.bbox_transform import bbox_transform_inv, clip_boxes
from Lib.nms_wrapper import nms
import numpy as np


#def im_detect_old(sess, fast_rcnn, im):
#    '''
#    Detect objects classes in an image
#    
#    sess: TensorFlow session
#    net: fast-rcnn network capable of producing rois, bbox_pred, cls_score
#    im: (tf.placeholder) image to test
#    '''
#    # Perform forward pass through Faster R-CNN network
#    feed_dict={net.x: im, net.im_dims: im.shape}
#    cls_prob, bbox_deltas, rois = sess.run([net.cls_prob, net.bbox_pred, net.rois],
#                                         feed_dict=feed_dict)
#
#    # Bounding boxes proposed by RPN    
#    boxes = rois[:,1:]
#
#    # Apply bounding box regression to RPN proposed boxes
#    pred_boxes = bbox_transform_inv(boxes, bbox_deltas)
#    pred_boxes = clip_boxes(pred_boxes, im.shape)
#    
#    return cls_prob, pred_boxes
    
    
def im_detect(model, inputs, key):
    '''
    Detect objects in an input image using the model
    
    model: faster_rcnn model object
    inputs: [0] The image to perform detections on
            [1] Ground-truth boxes. Should be None, since this is evaluation
            [2] Image dimensions 
    key: the network to evaluate on (eg. VALID or TEST)
    '''
    # Graph Inputs for Detection
    feed_dict = {model.x[key]: inputs[0], model.gt_boxes[key]: inputs[1], model.im_dims[key]: inputs[2]}
    
    # Graph Outputs for Detection
    cls_prob_out = model.fast_rcnn_net[key].get_cls_prob()
    bbox_ref_out = model.fast_rcnn_net[key].get_bbox_refinement()
    rois_out     = model.roi_proposal_net[key].get_rois()
    
    # Evaluate the graph
    cls_prob, bbox_deltas, rois = model.sess.run([cls_prob_out, bbox_ref_out, rois_out], feed_dict)  
    
    # Bounding boxes proposed by RPN    
    boxes = rois[:,1:]

    # Apply bounding box regression to RPN proposed boxes
    pred_boxes = bbox_transform_inv(boxes, bbox_deltas)
    pred_boxes = clip_boxes(pred_boxes, inputs[2])
    
    return cls_prob, pred_boxes
    
def vis_detections(im, class_name, dets, thresh=0.3):
    """Visual debugging of detections."""
    import matplotlib.pyplot as plt
    im = im[:, :, (2, 1, 0)]
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = dets[i, :4]
        score = dets[i, -1]
        if score > thresh:
            plt.cla()
            plt.imshow(im)
            plt.gca().add_patch(
                plt.Rectangle((bbox[0], bbox[1]),
                              bbox[2] - bbox[0],
                              bbox[3] - bbox[1], fill=False,
                              edgecolor='g', linewidth=3)
                )
            plt.title('{}  {:.3f}'.format(class_name, score))
            plt.show()
    
    
def apply_nms(all_boxes, thresh):
    """
    Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    for cls_ind in range(num_classes):
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == []:
                continue
            # CPU NMS is much faster than GPU NMS when the number of boxes
            # is relative small (e.g., < 10k)
            # TODO(rbg): autotune NMS dispatch
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes    
    
    

    
    