#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 13:28:34 2017

@author: Kevin Liang

Tests
"""

import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np

from Networks.convnet import convnet
from Networks.resnet import resnet
from Networks.faster_rcnn_networks import rpn, roi_proposal, fast_rcnn

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

class faster_rcnn_tests():
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 128, 128, 3], name='x')
        self.sess = tf.InteractiveSession()
        
    def test_all(self):
        self.test_convnet_dims()
        self.test_resnet_dims()
        
    def test_convnet_dims(self):
        filter_sizes = (3,3,3,3)
        output_channels = (32,64,64,128)
        strides = (1,2,1,2)
        cnn = convnet(self.x, filter_sizes, output_channels, strides)
        featureMaps = cnn.get_output()
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
                
        test_image = np.random.randint(0,256,[1,128,128,3])
        feat_val = self.sess.run(featureMaps,feed_dict={self.x:test_image})
        feat_val = np.array(feat_val)
        
        print(feat_val.shape)
        assert np.all(feat_val.shape == np.array([1,32,32,128]))
    
    def test_resnet_dims(self):
        cnn = resnet(50, self.x)
        featureMaps = cnn.get_output()
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        test_image = np.random.randint(0,256,[1,128,128,3])
        feat_val = self.sess.run(featureMaps,feed_dict={self.x:test_image})
        feat_val = np.array(feat_val)
        
        print(feat_val.shape)
        assert np.all(feat_val.shape == np.array([1,4,4,2048]))
    

        
        
def main():
    print("Initiating Tests")
    tester = faster_rcnn_tests()
    tester.test_all()

if __name__ == "__main__":
    main()