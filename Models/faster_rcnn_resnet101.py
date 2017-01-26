# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 13:22:36 2016

@author: Kevin Liang

Faster R-CNN model using ResNet as the convolutional feature extractor
"""

import sys
sys.path.append('../')

from Lib.TensorBase.tensorbase.base import Model, Data

from Development.tests import faster_rcnn_tests

#from Networks.resnet import resnet
from Networks.convnet import convnet
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
    'num_classes': 11,   # 10 digits, +1 for background
    'batch_size': 1,
    'display_step': 200,
    'weight_decay': 1e-7,
    'lr_decay': 0.999,
    'num_epochs': 10,
    'lr_iters': [(5e-3, 5000), (5e-3, 7500), (5e-4, 10000), (5e-5, 10000)],
    'anchor_scales': [1,2,3]
}

class faster_rcnn_resnet101(Model):
    def __init__(self, flags_input, run_num):
        super().__init__(flags_input, run_num, vram=0.9)
        self.print_log("Seed: %d" % flags['seed'])
        
    def _data(self):
        file = '/home/dcs41/Documents/tf-Faster-RCNN/Data/data_clutter/clutter_mnist_train.tfrecords'
        self.x, self.gt_boxes, self.im_dims = Data.batch_inputs(self.read_and_decode, file, batch_size=self.flags['batch_size'])
        self.num_train_images = 55000
        
    def _summaries(self):
        ''' Define summaries for TensorBoard '''
        tf.summary.scalar("Total_Loss", self.cost)
        tf.summary.scalar("RPN_cls_Loss", self.rpn_cls_loss)
        tf.summary.scalar("RPN_bbox_Loss", self.rpn_bbox_loss) 
        tf.summary.scalar("Fast_RCNN_cls_Loss", self.fast_rcnn_cls_loss)
        tf.summary.scalar("Fast_RCNN_bbox_Loss", self.fast_rcnn_bbox_loss)
        tf.summary.image("x", self.x)
        
    def _network(self):
        ''' Define the network outputs '''
        # Convolutional Feature Extractor: ResNet101
#        self.cnn = resnet(101, self.x)
        # Simpler convnet for debugging
        self.cnn = convnet(self.x, [5, 3, 3, 3, 3, 3], [64, 96, 128, 172, 256, 512], strides=[2, 2, 2, 2, 2, 2])
        featureMaps = self.cnn.get_output()
        _feat_stride = self.cnn.get_feat_stride()
        
        # Region Proposal Network (RPN)
        self.rpn_net = rpn(featureMaps, self.gt_boxes, self.im_dims, _feat_stride, flags)
        
        rpn_cls_score = self.rpn_net.get_rpn_cls_score()
        rpn_bbox_pred = self.rpn_net.get_rpn_bbox_pred()
        
        # ROI proposal
        self.roi_proposal_net = roi_proposal(rpn_cls_score, rpn_bbox_pred, self.gt_boxes, self.im_dims, flags)
        
        # R-CNN Classification
        self.fast_rcnn_net = fast_rcnn(featureMaps, self.roi_proposal_net)
        
        
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
        
    def _run_train_iter(self):
        """ Run training iteration"""
        summary, _ = self.sess.run([self.merged, self.optimizer])
        return summary

    def _run_train_metrics_iter(self):
        """ Run training iteration with metrics output """
        summary, self.loss, _ = self.sess.run([self.merged, self.cost, self.optimizer])
        return summary        

    def test_print_image(self):
        """ Read data through self.sess and plot out """
        threads, coord = Data.init_threads(self.sess)  # Begin Queues
        print("Running 100 iterations of simple data transfer from queue to np.array")
        for i in range(100):
            x, gt_boxes = self.sess.run([self.x, self.gt_boxes])
            print(i)
        # Plot an example
        faster_rcnn_tests.plot_img(x[0], gt_boxes[0])
        Data.exit_threads(threads, coord)  # Exit Queues

    def train(self):
        """ Run training function. Save model upon completion """
        iterations = int(np.ceil(self.num_train_images/self.flags['batch_size']) * self.flags['num_epochs'])
        threads, coord = Data.init_threads(self.sess)  # Begin Queues
        self.print_log('Training for %d iterations' % iterations)
        for i in range(iterations):
            if self.step % self.flags['display_step'] != 0:
                summary = self._run_train_iter()
            else:
                summary = self._run_train_metrics_iter()
                self._record_train_metrics()
            self._record_training_step(summary)
            print(self.step)
        self._save_model(section=1)
        Data.exit_threads(threads, coord)  # Exit Queues

    def _record_train_metrics(self):
        """ Record training metrics """
        self.print_log('Step %d: loss = %.6f' % (self.step, self.loss))

    @staticmethod
    def read_and_decode(example_serialized):
        """ Read and decode binarized, raw MNIST dataset from .tfrecords file generated by clutterMNIST.py """
        features = tf.parse_single_example(
            example_serialized,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'gt_boxes': tf.FixedLenFeature([5], tf.int64, default_value=[-1]*5),  # 10 classes in MNIST
                'dims': tf.FixedLenFeature([2], tf.int64, default_value=[-1]*2)
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
    print("Survived")

if __name__ == "__main__":
    main()