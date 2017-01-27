# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 13:22:36 2016

@author: Kevin Liang

Faster R-CNN model using ResNet as the convolutional feature extractor

Reorganizing a few things relative to faster_rcnn_conv5
"""

import sys

sys.path.append('../')

from Lib.TensorBase.tensorbase.base import Model, Data
from Lib.test_aux import im_detect, vis_detections, apply_nms

from Development.tests import faster_rcnn_tests

from Networks.convnet import convnet
from Networks.faster_rcnn_networks import rpn, roi_proposal, fast_rcnn

import numpy as np
import tensorflow as tf

# Global Dictionary of Flags
flags = {
    'save_directory': './',
    'model_directory': 'conv5/',
    'restore': False,
    'batch_size': 1,
    'display_step': 200,
    'num_epochs': 50,
    'num_classes': 11,   # 10 digits, +1 for background
    'anchor_scales': [1,2,3]
}


class faster_rcnn_resnet101(Model):
    def __init__(self, flags_input, run_num):
        super().__init__(flags_input, run_num, vram=0.3)
        self.print_log("Seed: %d" % flags['seed'])
        self.threads, self.coord = Data.init_threads(self.sess)

    def _data(self):
        file_train = '/home/dcs41/Documents/tf-Faster-RCNN/Data/data_clutter/clutter_mnist_train.tfrecords'
        self.x['TRAIN'], self.gt_boxes['TRAIN'], self.im_dims['TRAIN'] = Data.batch_inputs(self.read_and_decode, file_train,
                                                                batch_size=self.flags['batch_size'])
        
        file_valid = '/home/dcs41/Documents/tf-Faster-RCNN/Data/data_clutter/clutter_mnist_valid.tfrecords'
        self.x['VALID'], self.gt_boxes['VALID'], self.im_dims['VALID']= Data.batch_inputs(self.read_and_decode, file_valid, mode="eval",
                                                                batch_size=self.flags['batch_size'])
        
        self.x['TEST']        = tf.placeholder(tf.float32,[None,128,128,1])
        self.gt_boxes['TEST'] = tf.placeholder(tf.int32,[5])
        self.im_dims['TEST']  = tf.placeholder(tf.int32,[2])        
        
        self.num_train_images = 55000
        self.num_valid_images = 5000
        self.num_test_images  = 10000

    def _summaries(self):
        ''' Define summaries for TensorBoard '''
        tf.summary.scalar("Total_Loss", self.cost)
        tf.summary.scalar("RPN_cls_Loss", self.rpn_cls_loss)
        tf.summary.scalar("RPN_bbox_Loss", self.rpn_bbox_loss)
        tf.summary.scalar("Fast_RCNN_Cls_Loss", self.fast_rcnn_cls_loss)
        tf.summary.scalar("Fast_RCNN_Bbox_Loss", self.fast_rcnn_bbox_loss)
        tf.summary.image("x_train", self.x['TRAIN'])

    def _network(self):
        ''' Define the network outputs '''
        # Train network
        with tf.variable_scope('model'):
            self._faster_rcnn(self.x['TRAIN'], self.gt_boxes['TRAIN'], self.im_dims['TRAIN'], 'TRAIN')

        # Valid network => Uses same weights as train network
        with tf.variable_scope('model', reuse=True):
            assert tf.get_variable_scope().reuse is True
            self._faster_rcnn(self.x['VALID'], self.gt_boxes['VALID'], self.im_dims['VALID'], 'VALID')
            
        # Test network => Uses same weights as train network
        with tf.variable_scope('model', reuse=True):
            assert tf.get_variable_scope().reuse is True
            self._faster_rcnn(self.x['TEST'], self.gt_boxes['TEST'], self.im_dims['TEST'], 'TEST')

    def _faster_rcnn(self, x, gt_boxes, im_dims, key):
            # TODO: Switch to Layer convnet or resnet
            self.cnn[key] = convnet(x, [5, 3, 3, 3, 3], [64, 96, 128, 172, 256], strides=[2, 2, 2, 2, 2])
            featureMaps = self.cnn[key].get_output()
            _feat_stride = self.cnn[key].get_feat_stride()

            # Region Proposal Network (RPN)
            self.rpn_net[key] = rpn(featureMaps, gt_boxes, im_dims, _feat_stride, flags)

            # Roi Pooling
            self.roi_proposal_net[key] = roi_proposal(self.rpn_net[key], gt_boxes, im_dims, key, flags)

            # R-CNN Classification
            self.fast_rcnn_net[key] = fast_rcnn(featureMaps, self.roi_proposal_net[key])
            
    def _optimizer(self):
        ''' Define losses and initialize optimizer '''
        # Losses
        self.rpn_cls_loss = self.rpn_net.get_rpn_cls_loss()
        self.rpn_bbox_loss = self.rpn_net.get_rpn_bbox_loss()
        self.fast_rcnn_cls_loss = self.fast_rcnn_net.get_fast_rcnn_cls_loss()
        self.fast_rcnn_bbox_loss = self.fast_rcnn_net.get_fast_rcnn_bbox_loss()

        # Total Loss
        self.cost = tf.reduce_sum(self.rpn_cls_loss + self.rpn_bbox_loss + self.fast_rcnn_cls_loss + self.fast_rcnn_bbox_loss)

        # Optimization operation
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)

    def test_print_image(self):
        """ Read data through self.sess and plot out """
        threads, coord = Data.init_threads(self.sess)  # Begin Queues
        print("Running 100 iterations of simple data transfer from queue to np.array")
        for i in range(100):
            x, gt_boxes = self.sess.run([self.x['TRAIN'], self.gt_boxes['TRAIN']])
            print(i)
        # Plot an example
        faster_rcnn_tests.plot_img(x[0], gt_boxes[0])
        Data.exit_threads(threads, coord)  # Exit Queues

    def _run_train_iter(self):
        """ Run training iteration"""
        summary, _ = self.sess.run([self.merged, self.optimizer])
        return summary

    def _record_train_metrics(self):
        """ Record training metrics """
        loss = self.sess.run(self.cost)
        self.print_log('Step %d: loss = %.6f' % (self.step, loss))

    def train(self):
        """ Run training function. Save model upon completion """
        epochs = 0
        iterations = int(np.ceil(self.num_train_images/self.flags['batch_size']) * self.flags['num_epochs'])
        self.print_log('Training for %d iterations' % iterations)
        for i in range(iterations):
            summary = self._run_train_iter()
            if self.step % self.flags['display_step'] == 0:
                self._record_train_metrics()
                bbox, cls = self.sess.run([self.fast_rcnn_net.get_bbox_refinement(), self.fast_rcnn_net.get_cls_score()])
                print(bbox.shape)
                print(cls.shape)
            if self.step % (self.flags['num_epochs'] * self.num_train_images) == 0:
                self._save_model(section=epochs)
                epochs += 1
            self._record_training_step(summary)
            print(self.step)
        Data.exit_threads(self.threads, self.coord)  # Exit Queues

    def test(self): 
        """ Evaluate network on the test set. """
        num_images = self.num_test_images
        self.print_log('Testing %d images' % num_images)
#        test_list = 
        
        for i in range(num_images):
            
            inputs = []
            cls_probs,boxes = im_detect(self, inputs, 'TEST')

    @staticmethod
    def read_and_decode(example_serialized):
        """ Read and decode binarized, raw MNIST dataset from .tfrecords file generated by clutterMNIST.py """
        features = tf.parse_single_example(
            example_serialized,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'gt_boxes': tf.FixedLenFeature([5], tf.int64, default_value=[-1] * 5),  # 10 classes in MNIST
                'dims': tf.FixedLenFeature([2], tf.int64, default_value=[-1] * 2)
            })
        # now return the converted data
        gt_boxes = features['gt_boxes']
        dims = features['dims']
        image = tf.decode_raw(features['image'], tf.float32)
        image = tf.reshape(image, [128, 128, 1])
        return image, tf.cast(gt_boxes, tf.int32), tf.cast(dims, tf.int32)

        
def main():
    flags['seed'] = 1234
    model = faster_rcnn_resnet101(flags, run_num=1)
    model.train()


if __name__ == "__main__":
    main()