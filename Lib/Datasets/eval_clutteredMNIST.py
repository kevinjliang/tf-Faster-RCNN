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


def cluttered_mnist_eval(test_image_object, test_directory, num_images, ovthresh=0.5):
    """
    Evalulates predicted detections on cluttered MNIST dataset
    :param test_image_object: array, obj[cls][image] = N x 5 [x1, y1, x2, y2, cls_score]
    :param test_directory: str, location of the "Test" folder. Should end with "../Test/".
    :param ovthresh: float, between 1 and 0, threshold for rejecting bbox
    :return: class_metrics: list, each index is a digit class which holds a tuple of rec, prec, and ap
    """
    # Get Ground Truth numbers for classes
    total_num = np.zeros([11])
    print('Loading Grouth Truth Data to count number of grouth truth per class')
    for x in tqdm(range(num_images)):  # number of test data
        key = 'img' + str(x)
        gt_boxes = np.loadtxt(test_directory + 'Annotations/' + key + '.txt', ndmin=2)
        for g in range(gt_boxes.shape[0]):
            label = int(gt_boxes[g, 4])
            total_num[label] += 1

    # Designate arrays to hold ap for each class
    class_metrics = list()

    # Calculate IoU for all classes and all images
    for c in range(1, len(test_image_object)):  # loop through all classes (skip background class)
        print('Now Calculating average precision for class: %d' % c)
        class_tp = list()
        class_fp = list()

        # go down dets and mark TPs and FPs
        for i in range(len(test_image_object[c])):  # loop through all images

            # Get image detections and preallocate arrays
            image_dets = test_image_object[c][i]
            nd = len(image_dets)
            tp = np.zeros(nd)
            fp = np.zeros(nd)

            # Get groundtruth
            key = 'img' + str(i)
            gt_boxes = np.loadtxt(test_directory + '/Annotations/' + key + '.txt', ndmin=2)

            bbgt = gt_boxes[:, :4]
            labels = gt_boxes[:, 4]
            labels_det = [False] * len(labels)
            ovmax = -np.inf  # In case no overlaps result

            for d in range(nd):  # loop through all dets in a given image

                # Store particular dets as bb
                bb = image_dets[d, :4]

                # compute overlaps intersection
                ixmin = np.maximum(bbgt[:, 0], bb[0])
                iymin = np.maximum(bbgt[:, 1], bb[1])
                ixmax = np.minimum(bbgt[:, 2], bb[2])
                iymax = np.minimum(bbgt[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (bbgt[:, 2] - bbgt[:, 0] + 1.) *
                       (bbgt[:, 3] - bbgt[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

                # Threshold
                if ovmax > ovthresh:
                    if not labels_det[jmax]:
                        tp[d] = 1.
                        labels_det[jmax] = True
                    else:
                        fp[d] = 1.
                else:
                    fp[d] = 1.

            # Add scores from all dets in one image to the class true positives and false positives
            class_tp.extend(tp)
            class_fp.extend(fp)

        # compute precision recall
        fp = np.cumsum(class_fp)
        tp = np.cumsum(class_tp)
        rec = tp / float(total_num[c])

        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec)
        class_metrics.append(ap)
    print('Mean Average Precision: %f' % np.mean(class_metrics))
    return class_metrics
