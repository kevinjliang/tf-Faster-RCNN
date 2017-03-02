#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 12:19:01 2017

@author: Daniel Salo (Modifications)

Functions for testing Faster RCNN net on Cluttered MNIST and getting Mean Average Precision
"""

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
from tqdm import tqdm


def voc_ap(rec, prec):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def cluttered_mnist_eval(test_image_object, data_directory, num_images, ovthresh=0.4):
    """
    Evalulates predicted detections on cluttered MNIST dataset
    :param test_image_object: array, obj[cls][image] = N x 5 [x1, y1, x2, y2, cls_score]
    :param data_directory: str, location of the evaluation folder. Should end with "../Test/" or "../Valid/".
    :param ovthresh: float, between 1 and 0, threshold for rejecting bbox
    :return: class_metrics: list, each index is a digit class which holds a tuple of rec, prec, and ap
    """
    # Get Ground Truth numbers for classes
    total_num = np.zeros([11])
    print('Loading Ground Truth Data to count number of ground truth per class')
    for x in tqdm(range(num_images)):  # number of test data
        key = 'img' + str(x)
        gt_boxes = np.loadtxt(data_directory + 'Annotations/' + key + '.txt', ndmin=2)
        for g in range(gt_boxes.shape[0]):
            label = int(gt_boxes[g, 4])
            total_num[label] += 1
    print('Total Number of Images per class:')
    print(total_num)

    # Define class_metrics list
    # Labels array holds booleans as to whether an image/gt has been counted yet
    class_metrics = list()
    labels = [False] * np.sum(total_num).astype(int)

    # Calculate IoU for all classes and all images
    for c in range(1, len(test_image_object)):  # loop through all classes (skip background class)

        # Transform test_image_object into an np.array with all dets together.
        class_dets = test_image_object[c]
        all_dets = list()
        for dets_list in class_dets:
            if len(dets_list) == 0:
                continue
            else:
                all_dets.extend(dets_list)
        all_dets = np.array(all_dets)

        # Sort the detections by confidence (cls_score)
        confidence = [score for score in all_dets[:, 0]]
        confidence = np.array(confidence)
        indx = np.argsort(-confidence)
        all_dets = all_dets[indx, :]

        # Preallocate true positive and false positive arrays with zeros (number of detections)
        nd = confidence.shape[0]
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        # go down dets and mark TPs and FPs
        for d in range(nd):  # loop through all detections

            # Get ground truth
            img_indx = int(all_dets[d, 1])
            key = 'img' + str(img_indx)
            gt_boxes = np.loadtxt(data_directory + '/Annotations/' + key + '.txt', ndmin=2)

            # Store proposal dets as bb and ground truth as bbgt
            bbgt = gt_boxes[:, :4]
            bb = all_dets[d, 2:6]

            # compute intersection
            ixmin = np.maximum(bbgt[:, 0], bb[0])
            iymin = np.maximum(bbgt[:, 1], bb[1])
            ixmax = np.minimum(bbgt[:, 2], bb[2])
            iymax = np.minimum(bbgt[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # compute union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (bbgt[:, 2] - bbgt[:, 0] + 1.) *
                   (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

            # computer IoU
            overlaps = inters / uni
            ovmax = np.max(overlaps)

            # Threshold
            if ovmax > ovthresh:
                if labels[img_indx] is False:  # ensure no ground truth box is double counted
                    tp[d] = 1
                    labels[img_indx] = True
                else:
                    fp[d] = 1
            else:
                fp[d] = 1.

        # compute recall and precision
        cum_fp = np.cumsum(fp)
        cum_tp = np.cumsum(tp)
        rec = cum_tp / float(total_num[c])
        prec = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)  # avoid divide by zero

        # compute average precision and store
        ap = voc_ap(rec, prec)
        class_metrics.append(ap)
    print('Mean Average Precision: %f' % np.mean(class_metrics))
    return class_metrics
