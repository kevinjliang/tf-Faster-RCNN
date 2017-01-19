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

from Networks.resnet import resnet
#from Networks.faster_rcnn_networks import rpn, roi_proposal, fast_rcnn

class faster_rcnn_tests():
    def __init__(self):
        self.x = tf.placeholder(tf.float32, [None, 128, 128, 3], name='x')
        self.sess = tf.InteractiveSession()
        
    def test_all(self):
        self.test_resnet_dims()
    
    def test_resnet_dims(self):
        cnn = resnet(50, self.x)
        featureMaps = cnn.get_output()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
        test_image = np.random.randint(0,256,[1,128,128,3])
        feat_val = self.sess.run([featureMaps],feed_dict={self.x:test_image})
        feat_val = np.array(feat_val)
        
        print(feat_val.shape)
        assert np.all(feat_val.shape == [1,4,4,2048])
    
        
def main():
    print("Initiating Tests")
    tester = faster_rcnn_tests()
    tester.test_all()

if __name__ == "__main__":
    main()