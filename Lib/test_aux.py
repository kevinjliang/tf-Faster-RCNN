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

    
import matplotlib.pyplot as plt
import cv2
from Lib.fast_rcnn_config import cfg, get_output_dir
import Pickle
import os
    
def test_net(sess, net, imdb, weights_filename , max_per_image=300, thresh=0.05, vis=False):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    output_dir = get_output_dir(imdb, weights_filename)
    # timers
#    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    for i in range(num_images):
        box_proposals = None

        im = cv2.imread(imdb.image_path_at(i))
#        _t['im_detect'].tic()
        scores, boxes = im_detect(sess, net, im, box_proposals)
#        _t['im_detect'].toc()

#        _t['misc'].tic()
        if vis:
            image = im[:, :, (2, 1, 0)]
            plt.cla()
            plt.imshow(image)

        # skip j = 0, because it's the background class
        for j in range(1, imdb.num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            cls_scores = scores[inds, j]
            cls_boxes = boxes[inds, j*4:(j+1)*4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            if vis:
                vis_detections(image, imdb.classes[j], cls_dets)
            all_boxes[j][i] = cls_dets
        if vis:
           plt.show()
        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
#        _t['misc'].toc()
#
#        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
#              .format(i + 1, num_images, _t['im_detect'].average_time,
#                      _t['misc'].average_time))

    det_file = os.path.join(output_dir, 'detections.pkl')
    with open(det_file, 'wb') as f:
        Pickle.dump(all_boxes, f)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, output_dir)
