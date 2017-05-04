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

from .bbox_transform import clip_boxes, bbox_transform_inv
from .evaluate_predictions import evaluate_predictions, compute_iou
from .faster_rcnn_config import cfg
from .image_functions import image_preprocessing, vis_preprocessing, read_image
from .nms_wrapper import nms

import matplotlib
matplotlib.use('TkAgg')  # For Mac OS
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import pickle
from tqdm import tqdm


def test_net(data_directory, names, sess, tf_inputs, tf_outputs, max_per_image=300, key='valid', thresh=0.1, vis=False):
    """Test a Fast R-CNN network on an image database.
    
    data_directory: Directory to the data. Should be organized as follows:
        |--data_directory/
            |--Annotations/
                |--*.txt (Annotation Files: (x1,y1,x2,y2,l))
            |--Images/
                |--*.[png/jpg] (Image files)
            |--Names/
                |--[train/valid/test].txt (List of data)
    names: list of data files (contents of the above [train/valid/test].txt file)             
    sess: TensorFlow session  
    tf_inputs: TensorFlow tensor inputs to the computation graph
            [0] x: the image input
            [1] im_dims: image dimensions of input (height, width)
    tf_outputs: TensorFlow tensor outputs of the computation graph
            [0] rois: RoIs produced by the RPN
            [1] cls_prob: Classifier probabilities of each object by the RCNN
            [2] bbox_ref: Bounding box refinements by the RCNN 
    max_per_image: Maximum number of detections per image (TODO: currently does nothing)
    thresh: Threshold for a non-background detection to be counted
    vis: Visualize detection in figure using vis_detections
            
    """

    # Create output directory
    det_dir = data_directory + 'Outputs/'
    if not os.path.exists(det_dir):
        os.makedirs(det_dir)

    # Either detect boxes or load previous detections  
    all_boxes, detection_made = _detect_boxes(data_directory, names, sess, tf_inputs, tf_outputs, thresh, det_dir, key, vis)

    # Ensure that at least some detections were made
    if not detection_made:
        print("No detections were made")
        return [0.0]*cfg.NUM_CLASSES

    if cfg.TEST.GROUNDTRUTH:
        class_metrics = evaluate_predictions(all_boxes, data_directory, names)
        return class_metrics
    else:
        return [0.0]*cfg.NUM_CLASSES


def _detect_boxes(data_directory, names, sess, tf_inputs, tf_outputs, thresh, det_dir, key, vis):
    """ Detection of bounding boxes in test image. See test_net for detailed description of inputs """
    
    num_images = len(names)
    
    det_file = det_dir + key + '_detections.pkl'
    
    # Check for detections already. If they don't exists or if user wants to overwrite, then detect boxes from scratch
    if os.path.exists(det_file) and key == 'TEST':
        overwrite = input('Saved detections were detected. Would you like to overwrite? [y/n]')
        if overwrite.strip().lower() == 'n':
            # Load boxes from pickle file
            print('Loading boxes from %s' % det_file)
            detection_made = True
            all_boxes = pickle.load(open(det_file, "rb"))
            return all_boxes, detection_made
        else:
            assert overwrite.strip().lower() == 'y', 'Invalid input'

    # Ensure at least one detection is made or else an error will be thrown.
    detection_made = False

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, prob)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(cfg.NUM_CLASSES)]

    # Loop through all images and generate proposals
    for i in tqdm(range(num_images)):

        # Read in file
        im_file = data_directory + 'Images/' + names[i] + cfg.IMAGE_FORMAT
        image = read_image(im_file)

        # Perform Detection
        probs, boxes = _im_detect(sess, image, tf_inputs, tf_outputs)
        
        # Collect all detections for if visualizing
        if vis:
            dets = list()
            cls = list()

        # skip j = 0, because it's the background class
        for j in range(1, cfg.NUM_CLASSES):
            inds = np.where(probs[:, j] > thresh)[0]
            if len(inds) == 0:
                continue
            detection_made = True
            cls_probs = probs[inds, j]                  # Class Probabilities
            cls_boxes = boxes[inds, j * 4:(j + 1) * 4]  # Class Box Predictions
            cls_dets = np.hstack((cls_boxes, cls_probs[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(cls_dets, cfg.TEST.NMS)          # Apply NMS
            cls_dets = cls_dets[keep, :]
            all_boxes[j][i] = cls_dets

            if vis:
                dets.extend(cls_dets)
                cls.extend(np.repeat(j, cls_dets.shape[0]))

        if vis:
            # Formating for visualization
            outputfilename = det_dir + names[i] + cfg.IMAGE_FORMAT
            dets = np.array(dets)
            cls = np.array(cls)

            # Load image and GT information
            if cfg.TEST.GROUNDTRUTH:
                gt = np.loadtxt(data_directory + 'Annotations/' + names[i] + '.txt', ndmin=2)
            else:
                gt = None

            # Visualize detections
            _vis_detections(image, gt, dets, cls, outputfilename)

    # Save Detections
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f)
        
    return all_boxes, detection_made


def _im_detect(sess, image, tf_inputs, tf_outputs):
    """
    Detect objects in an input image using the model

    sess: TensorFlow session
    image: Image to perform detection on. Should be numpy array
    tf_inputs: TensorFlow tensor inputs to the computation graph
            [0] x: the image input (rows, cols, channels)
            [1] im_dims: image dimensions of input (height, width)
    tf_outputs: TensorFlow tensor outputs of the computation graph
            [0] rois: RoIs produced by the RPN
            [1] cls_prob: Classifier probabilities of each object by the RCNN
            [2] bbox_ref: Bounding box refinements by the RCNN
    """

    # Applies dataset specific pre-processing to image
    image = image_preprocessing(image)

    im_dims = np.array(image.shape[1:3]).reshape([1, 2])

    feed_dict = {tf_inputs[0]: image, tf_inputs[1]: im_dims}

    # Evaluate the graph
    rois, cls_prob, bbox_deltas = sess.run(tf_outputs, feed_dict)

    # Bounding boxes proposed by Faster RCNN
    boxes = rois[:, 1:]

    if cfg.TEST.BBOX_REFINE:
        # Apply bounding box regression to Faster RCNN proposed boxes
        pred_boxes = bbox_transform_inv(boxes, bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, np.squeeze(im_dims))
    else:
        # Or just repeat the boxes, one for each classe
        pred_boxes = clip_boxes(boxes, np.squeeze(im_dims))
        pred_boxes = np.tile(pred_boxes, (1, bbox_deltas.shape[1]))

    return cls_prob, pred_boxes


def _vis_detections(im, gt_boxes, dets, cls, filename=None, skip_background=True):
    """Visual debugging of detections."""
    # Perform preprocessing (or not)
    im = vis_preprocessing(im)
    
    # Plot image
    fig, ax = plt.subplots(1)
    
    if len(im.shape) == 3 and cfg.TEST.CMAP != 'jet':
        ax.imshow(im[:,:,0], cmap=cfg.TEST.CMAP)    
    else:
        ax.imshow(im, cmap=cfg.TEST.CMAP)

    if gt_boxes is not None:    
        # Extract ground truth classes and boxes
        cls_gt = gt_boxes[:, 4]
        bb_gt = gt_boxes[:, :4]
    
        # Plot ground truth boxes
        if cfg.TEST.PLOT_GROUNDTRUTH:
            for g in range(gt_boxes.shape[0]):
                _plot_patch(ax, bb_gt[g, :], None, None, 'g')

    # Plot detections
    for i in range(dets.shape[0]):

        # Extract class probability and bounding box
        prob = dets[i, 4]
        bb = dets[i, :4]

        if gt_boxes is not None:
            # Compute IOU with gt_boxes
            ovmax, ovargmax = compute_iou(bb=bb, bbgt=bb_gt)
    
            # Determine correct color of box
            if cls[i] == cls_gt[ovargmax] and ovmax > 0.5:
                color = 'b'  # Correct label
            elif cls[i] == 0:
                color = 'm'  # Background label
                if skip_background:
                    continue
            else:
                color = 'r'  # Mislabel   
        else:
            color = 'c'
            
        # Plot the rectangle
        class_name = cfg.CLASSES[cls[i]]
        _plot_patch(ax, bb, prob, class_name, color)

    # Save figure
    if filename is not None:
        fig.savefig(filename)
    # Display Final composite image
    else:
        plt.show()

    # Close figure
    plt.close(fig)


def _plot_patch(ax, bbox, prob, class_name, color):
    """ Plot a rectangle (labeled with color, class_name, and prob) on the test image """

    # Calculate Bounding Box Rectangle and plot it
    height = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]
    rect = patches.Rectangle((bbox[0], bbox[1]), width, height, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)

    # Add confidence prob and class text to box
    if prob is not None:
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, prob),
                bbox=dict(facecolor=color, alpha=0.5),
                fontsize=8, color='white')
