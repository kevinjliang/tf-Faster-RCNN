# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 13:22:36 2016

@author: Kevin Liang

Faster R-CNN model using ResNet as the convolutional feature extractor
"""

import sys
sys.path.append('../')

from Lib.TensorBase.tensorbase.base import Model
#from Lib.TensorBase.tensorbase.base import Layers
from Lib.TensorBase.tensorbase.data import Mnist

from Networks.resnet101 import resnet101
from Networks.faster_rcnn_networks import rpn, roi_proposal, fast_rcnn

import tensorflow as tf
import numpy as np

# Global Dictionary of Flags
flags = {
    'data_directory': '../Data/MNIST/',
    'save_directory': '../Logs/summaries/',
    'model_directory': 'resnet101/',
    'restore': False,
    'restore_file': 'start.ckpt',
    'datasets': 'MNIST',
    'image_dim': 28,
    'hidden_size': 10,
    'num_classes': 10,
    'batch_size': 100,
    'display_step': 200,
    'weight_decay': 1e-7,
    'lr_decay': 0.999,
    'lr_iters': [(5e-3, 5000), (5e-3, 7500), (5e-4, 10000), (5e-5, 10000)],
    'anchor_scales': [8,16,32]
}

class faster_rcnn_resnet101(Model):
    def __init__(self, flags_input, run_num):
        super().__init__(flags_input, run_num)
        self.print_log("Seed: %d" % flags['seed'])
        self.data = Mnist(flags_input)
        
    def _set_placeholders(self):
        self.x = tf.placeholder(tf.float32, [None, flags['image_dim'], flags['image_dim'], 1], name='x')
        self.y = tf.placeholder(tf.int32, shape=[1])
        self.gt_boxes = tf.placeholder(tf.int32, shape=[4]) 
        self.im_dims = tf.placeholder(tf.in32, shape=[2])
        
    def _set_summaries(self):
        ''' Define summaries for TensorBoard '''
        tf.summary.scalar("Total_Loss", self.cost)
        tf.summary.scalar("Weight_Decay_Loss", self.weight)
        tf.summary.image("x", self.x)
        # TODO add individual losses from RPN, Fast RCNN
        
    def _network(self):
        ''' Define the network outputs '''
        # Convolutional Feature Extractor: ResNet101
        self.cnn = resnet101(self.x)
        featureMaps = self.cnn.get_output()
        
        # Region Proposal Network (RPN)
        self.rpn_net = rpn(featureMaps,self.gt_boxes,self.im_dims,flags)
        
        rpn_cls_score = self.rpn_net.get_rpn_cls_score()
        rpn_bbox_pred = self.rpn_net.get_rpn_bbox_pred()
        
        # ROI proposal
        self.roi_proposal_net = roi_proposal(rpn_cls_score,rpn_bbox_pred,self.gt_boxes,self.im_dims,flags)
        
        rois = self.roi_proposal_net.get_rois()
        
        # R-CNN Classification
        self.fast_rcnn_net = fast_rcnn(featureMaps, rois, self.im_dims, flags)
        
    def _optimizer(self):
        ''' Define losses and initialize optimizer '''
        
        
        
        