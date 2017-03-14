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

from .bbox_transform import clip_boxes#, bbox_transform_inv
from .Datasets.eval_clutteredMNIST import cluttered_mnist_eval # Find a way to make this generalized
import matplotlib
matplotlib.use('TkAgg')  # For Mac OS
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import pickle
from tqdm import tqdm

    
def _im_detect(sess, image, tf_inputs, tf_outputs):
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
    if len(image.shape) == 2:
        image = np.expand_dims(np.expand_dims(image, 0), 3)
    else:
        image = np.expand_dims(image, 0)
    im_dims = np.array(image.shape[1:3]).reshape([1, 2])

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


def vis_detections(im, gt_boxes, dets, cls, data_info=None, skip_background=True):
    """Visual debugging of detections."""

    fig, ax = plt.subplots(1)

    # Plot image
    if len(im.shape)>2:
        im = np.squeeze(im[:,:,2])
    ax.imshow(im, cmap="gray")
    if dets.shape[0] > 0:
        plt.title(str(int(dets[0, 1]))) 
    
    # Plot ground truth box
    gt_cls = gt_boxes[0, 4]
    plot_patch(ax, gt_boxes[0][:4], None, None, 'g')
    
    # Plot detections
    for i in range(dets.shape[0]):
        score = dets[i, 0]
        bbox = dets[i, 2:]
        if cls[i] == gt_cls:
            color = 'b' # Correct label
        elif cls[i] == 0:
            color = 'm' # Background label
            if skip_background:
                continue
        else:
            color = 'r' # Mislabel
        
        if data_info == None:
            class_name = str(cls[i])
        else:
            class_name = data_info[2][cls[i]]
        plot_patch(ax, bbox, score, class_name, color)

    # Display Final composite image
    plt.show()


def plot_patch(ax, bbox, score, class_name, color):
    # Calculate Bounding Box Rectangle and plot it
    height = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]
    rect = patches.Rectangle((bbox[0], bbox[1]), width, height, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    
    # Add confidence score and class text to box
    if score is not None:
        ax.text(bbox[0], bbox[1] - 2,
            '{:s} {:.3f}'.format(class_name, score),
            bbox=dict(facecolor='blue', alpha=0.5),
            fontsize=8, color='white')
        

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
    detection_made = False
#    classes = data_info[2]
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
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
            detection_made = True
            cls_probs = probs[inds, j]              # Class Scores
            cls_index = np.repeat(i, len(inds))     # Index of Image
            cls_boxes = boxes[inds, j*4:(j+1)*4]    # Class Box Predictions
            cls_dets = np.hstack((cls_probs[:, np.newaxis], cls_index[:, np.newaxis], cls_boxes)) \
                .astype(np.float32, copy=False)
            all_boxes[j][i] = cls_dets

    # Ensure that at least some detections were made
    if not detection_made:
        print("No detections were made")
        return [[0]]

    if vis:
        for _ in range(num_images):
            i = input("Please select the index of the Test Image to display:")
            i = int(i)
            if i == -1:
                break
            
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
            vis_detections(image, gt, dets, cls, data_info)

    # Save detections
    det_dir = data_directory + 'Outputs/'
    if not os.path.exists(det_dir):
        os.makedirs(det_dir)
    
    det_file = det_dir + 'detections.pkl'
    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f)

    class_metrics = cluttered_mnist_eval(all_boxes, data_directory, num_images, num_classes)
        
    return class_metrics