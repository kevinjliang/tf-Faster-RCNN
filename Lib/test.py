#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 10:40:11 2017

@author: Kevin Liang (Modifications)
"""

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# TODO: Add num_classes to config file
# import sys
# sys.path.append('../')
# from fast_rcnn_config import cfg

def test_net(net, data, flags, max_per_image=100, thresh=0.05, vis=False):
    """
    Test a Faster R-CNN network on an image set.
    
    All detections are collected into:
        all_boxes[cls][image] = N x 5 array of detections in
        (x1, y1, x2, y2, score)
        
    TODO: Assuming data is 
    """
    
    num_images = data.shape[0]
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(flags['num_classes'])]
    
    # TODO put this output directory in config file
    output_dir = flags['save_directory'] + '../out'
    
    # TODO: add in timers
    
    # Loop through all images
    for i in range(num_images):
        box_proposals = None
        