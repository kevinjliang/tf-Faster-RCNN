#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 12:19:17 2017

@author: Kevin Liang

Basic ConvNet Architecture: Convolutional feature extractor

Support for arbitrary number of layers

Built on TensorBase: https://github.com/kevinjliang/TensorBase
"""
import sys
sys.path.append('../')

from Lib.TensorBase.tensorbase.base import Layers

import numpy as np
import tensorflow as tf


class convnet:    
    def __init__(self, x, filter_sizes, output_channels, strides=None):
        self.filter_sizes = filter_sizes
        self.depth = len(self.filter_sizes)
        self.output_channels = output_channels
        if strides is not None:
            self.strides = strides
        else:
            self.strides = np.ones([1,self.depth])
            
        self.network = self._network(x)
        
    def _network(self, x):
        conv_layers = Layers(x)
        
        # Make sure that number of layers is consistent
        assert len(self.output_channels) == self.depth
        assert len(self.strides) == self.depth
            
        # Convolutional layers
        scope = 'convnet' + str(self.depth)
        with tf.variable_scope(scope):
            for l in range(self.depth):
                conv_layers.conv2d(filter_size=self.filter_sizes[l], 
                                   output_channels=self.output_channels[l], 
                                   stride=self.strides[l],
                                   padding='SAME',
                                   b_value=None)
        
        return conv_layers
        
    def get_output(self):
        return self.network.get_output()
        
    def get_feat_stride(self):
        return np.prod(self.strides)
        