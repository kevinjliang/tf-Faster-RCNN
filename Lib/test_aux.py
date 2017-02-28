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

from .bbox_transform import bbox_transform_inv, clip_boxes
from .fast_rcnn_config import cfg
from .nms_wrapper import nms
from .Datasets.eval_clutteredMNIST import cluttered_mnist_eval # Find a way to make this generalized
#import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from scipy.misc import imread
from tqdm import tqdm

    
def _im_detect(sess, image, tf_inputs, tf_outputs, semi=0):
    '''
    Detect objects in an input image using the model
    
    sess: TensorFlow session
    image: Image to perform detection on. Should be numpy array    
    tf_inputs: TensorFlow tensor inputs to the computation graph
            [0] x: the image input
            [1] im_dims: image dimensions of input
    tf_outputs: TensorFlow tensor outputs of the computation graph
            [0] rois: RoIs produced by the RPN        
            [1] cls_prob: Classifier probabilities of each object by the RCNN
            [2] bbox_ref: Bounding box refinements by the RCNN 
    '''
    image = np.expand_dims(np.expand_dims(image, 0), 3)
    im_dims = np.array(image.shape[1:3]).reshape([1, 2])

    # Graph Inputs for Detection
    if semi == 1:
        feed_dict = {tf_inputs[0]: image, tf_inputs[1]: im_dims, tf_inputs[2]: 0}
    else:
        feed_dict = {tf_inputs[0]: image, tf_inputs[1]: im_dims}

    # Evaluate the graph
    rois, cls_prob, bbox_deltas = sess.run(tf_outputs, feed_dict)

    # Bounding boxes proposed by Faster RCNN
    boxes = rois[:, 1:]

    # Apply bounding box regression to Faster RCNN proposed boxes
#    pred_boxes = bbox_transform_inv(boxes, bbox_deltas)
#    pred_boxes = clip_boxes(pred_boxes, image.shape)

    boxes = clip_boxes(boxes, np.squeeze(im_dims))
    pred_boxes = np.tile(boxes,(1,bbox_deltas.shape[1]))
    
    return cls_prob, pred_boxes


def vis_detections(im, gt_boxes, dets, cls, skip_background=True):
    """Visual debugging of detections."""
    import matplotlib
    matplotlib.use('TkAgg')  # For Mac OS
    import matplotlib.patches as patches
    fig, ax = plt.subplots(1)

    gt_cls = gt_boxes[0, 4]
    for i in range(dets.shape[0]):
        bbox = dets[i, :]
        print(bbox)
        if cls[i] == gt_cls:
            color = 'b' # Correct label
        elif cls[i] == 0:
            color = 'm' # Background label
            if skip_background:
                continue
        else:
            color = 'r' # Mislabel
        ax.imshow(np.squeeze(im), cmap="gray")
        plot_patch(ax, patches, bbox, color)
    plt.title(str(int(gt_boxes[0, 4])))
    plot_patch(ax, patches, gt_boxes[0][:4], 'g')

    # Display Final composite image
    plt.show()


def plot_patch(ax, patches, bbox, color):
    # Calculate Bounding Box Rectangle and plot it
    height = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]
    rect = patches.Rectangle((bbox[0], bbox[1]), width, height, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)


def test_net(sess, data_directory, data_info, tf_inputs, tf_outputs, max_per_image=300, thresh=0.1, vis=False):
    """Test a Fast R-CNN network on an image database.
    
    sess: TensorFlow session  
    data_directory: Directory to the data. Should be organized as follows:
        |--data_directory/
            |--Annotations/
                |--*.txt (Annotation Files: (x1,y1,x2,y2,l))
            |--Images/
                |--*.png (Image files)
            |--ImageSets/
                |--test.txt (List of data)
    data_info: Information about the dataset
            [0] num_images: number of images to be tested
            [1] num_classes: number of classes in the dataset
            [2] classes: identities of each of the classes                
    tf_inputs: TensorFlow tensor inputs to the computation graph
            [0] x: the image input
            [1] im_dims: image dimensions of input
    tf_outputs: TensorFlow tensor outputs of the computation graph
            [0] rois: RoIs produced by the RPN
            [1] cls_prob: Classifier probabilities of each object by the RCNN
            [2] bbox_ref: Bounding box refinements by the RCNN 

    """
    num_images = data_info[0]
    num_classes = data_info[1]
#    classes = data_info[2]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    cls_dets = np.array([[1]])
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    print('Detecting boxes in images:')
    for i in tqdm(range(num_images)):
        # Read in file
        im_file = data_directory + 'Images/img' + str(i) + '.npy'  # Saved as numpy binary file
        image = np.load(im_file)
        
        # Perform Detection
        probs, boxes = _im_detect(sess, image, tf_inputs, tf_outputs)
    
        # skip j = 0, because it's the background class
        for j in range(1, num_classes):
            inds = np.where(probs[:, j] > thresh)[0]
            if len(inds) == 0:
                continue
            cls_probs = probs[inds, j]              # Class Scores
            cls_index = np.repeat(i, len(inds))     # Index of Image
            cls_boxes = boxes[inds, j*4:(j+1)*4]    # Class Box Predictions
            cls_dets = np.hstack((cls_probs[:, np.newaxis], cls_index[:, np.newaxis], cls_boxes)) \
                .astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets

            # Limit to max_per_image detections *over all classes*.
            if cls_dets.shape[0] > max_per_image > 0:
                    image_thresh = np.sort(cls_dets)[-max_per_image]
                    for j in range(1, num_classes):
                        keep = np.where(cls_dets[:, 0] >= image_thresh)[0]
                        all_boxes[j][i] = cls_dets[keep, :]

    if vis:
        for i in range(10):
            dets = list()
            cls = list()
            for c in range(1, num_classes):
                if len(all_boxes[c][i]) == 0:
                    continue
                else:
                    add = all_boxes[c][i]
                    dets.extend(add)
                    cls.extend(np.repeat(c, add.shape[0]))
            dets = np.array(dets)
            cls = np.array(cls)
            im_file = data_directory + 'Images/img' + str(i) + '.npy'  # Saved as numpy binary file
            image = np.load(im_file)
            gt = np.loadtxt(data_directory + 'Annotations/img' + str(i) + '.txt', ndmin=2)
            vis_detections(image, gt, dets[:, 2:], cls)

    # Save detections
    det_dir = data_directory + 'Outputs/'
    if not os.path.exists(det_dir):
        os.makedirs(det_dir)
    
    det_file = det_dir + 'detections.pkl'
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f)

    class_metrics = cluttered_mnist_eval(all_boxes, data_directory, num_images)
        
    return class_metrics

'''
def apply_nms(all_boxes, thresh):
    """Apply non-maximum suppression to all predicted boxes output by the
    test_net method.
    """
    num_classes = len(all_boxes)
    num_images = len(all_boxes[0])
    nms_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    for cls_ind in range(num_classes):
        for im_ind in range(num_images):
            dets = all_boxes[cls_ind][im_ind]
            if dets == list():
                continue
            keep = nms(dets, thresh, force_cpu=True)
            if len(keep) == 0:
                continue
            nms_boxes[cls_ind][im_ind] = dets[keep, :].copy()
    return nms_boxes
'''
#
#    
#import matplotlib.pyplot as plt
#import cv2
#from Lib.fast_rcnn_config import cfg, get_output_dir
#import Pickle
#import os
#
#    
#def test_net2(sess, net, imdb, weights_filename, max_per_image=300, thresh=0.05, vis=False):
#    """Test a Fast R-CNN network on an image database."""
#    num_images = len(imdb.image_index)
#    # all detections are collected into:
#    #    all_boxes[cls][image] = N x 5 array of detections in
#    #    (x1, y1, x2, y2, score)
#    all_boxes = [[[] for _ in range(num_images)]
#                 for _ in range(imdb.num_classes)]
#
#    output_dir = get_output_dir(imdb, weights_filename)
#    # timers
##    _t = {'im_detect' : Timer(), 'misc' : Timer()}
#
#    for i in range(num_images):
#        box_proposals = None
#
#        im = cv2.imread(imdb.image_path_at(i))
##        _t['im_detect'].tic()
#        scores, boxes = _im_detect(sess, net, im, box_proposals)
##        _t['im_detect'].toc()
#
##        _t['misc'].tic()
#        if vis:
#            image = im[:, :, (2, 1, 0)]
#            plt.cla()
#            plt.imshow(image)
#
#        # skip j = 0, because it's the background class
#        for j in range(1, imdb.num_classes):
#            inds = np.where(scores[:, j] > thresh)[0]
#            cls_scores = scores[inds, j]
#            cls_boxes = boxes[inds, j*4:(j+1)*4]
#            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
#                .astype(np.float32, copy=False)
#            keep = nms(cls_dets, cfg.TEST.NMS)
#            cls_dets = cls_dets[keep, :]
#            if vis:
#                _vis_detections(image, imdb.classes[j], cls_dets)
#            all_boxes[j][i] = cls_dets
#        if vis:
#           plt.show()
#        # Limit to max_per_image detections *over all classes*
#        if max_per_image > 0:
#            image_scores = np.hstack([all_boxes[j][i][:, -1]
#                                      for j in range(1, imdb.num_classes)])
#            if len(image_scores) > max_per_image:
#                image_thresh = np.sort(image_scores)[-max_per_image]
#                for j in range(1, imdb.num_classes):
#                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
#                    all_boxes[j][i] = all_boxes[j][i][keep, :]
##        _t['misc'].toc()
##
##        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
##              .format(i + 1, num_images, _t['im_detect'].average_time,
##                      _t['misc'].average_time))
#
#    det_file = os.path.join(output_dir, 'detections.pkl')
#    with open(det_file, 'wb') as f:
#        Pickle.dump(all_boxes, f)
#
#    print('Evaluating detections')
#    imdb.evaluate_detections(all_boxes, output_dir)
