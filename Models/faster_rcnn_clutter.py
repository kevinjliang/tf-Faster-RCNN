#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:13:11 2017

@author: kjl27
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 13:22:36 2016

@author: Kevin Liang

Faster R-CNN model using a simple 5-layer conv net as the convolutional feature extractor

Reorganizing a few things relative to faster_rcnn_conv5
"""

import sys

sys.path.append('../')

from Lib.TensorBase.tensorbase.base import Model, Data
from Lib.fast_rcnn_config import cfg_from_file
from Lib.test_aux import test_net#, vis_detections, _im_detect

from Networks.convnet import convnet
from Networks.faster_rcnn_networks_mnist import rpn, roi_proposal, fast_rcnn

import numpy as np
import tensorflow as tf
import argparse
import os
from tqdm import tqdm

# Global Dictionary of Flags
flags = {
    'data_directory': '../Data/data_clutter/',  # Location of training/testing files
    'save_directory': '../Logs/',  # Where to create model_directory folder
    'model_directory': 'conv5/',  # Where to create 'Model[n]' folder
    'batch_size': 1,  # This is fixed
    'display_step': 500,  # How often to display loss
    'num_classes': 11,  # 10 digits, +1 for background
    'classes': ('__background__', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0'),
    'anchor_scales': [0.5, 1, 2]
}


class FasterRcnnConv5(Model):
    def __init__(self, flags_input, dictionary):
        if flags_input['restore'] is True:
            self.epochs = flags_input['file_epoch']
        else:  # not restore
            self.epochs = 0
        self.lr = flags['learning_rate']    
        super().__init__(flags_input, flags_input['run_num'], vram=0.2, restore=flags_input['restore_num'])
        self.print_log(dictionary)
        self.print_log(flags_input)
        self.threads, self.coord = Data.init_threads(self.sess)

    def _data(self):
        # Initialize placeholder dicts
        self.x = {}
        self.gt_boxes = {}
        self.im_dims = {}

        # Train data
        file_train = flags['data_directory'] + 'clutter_mnist_train.tfrecords'
        self.x['TRAIN'], self.gt_boxes['TRAIN'], self.im_dims['TRAIN'] = Data.batch_inputs(self.read_and_decode,
                                                                                           file_train, batch_size=self.flags['batch_size'])
        # Validation data. No GT Boxes.
        self.x['EVAL'] = tf.placeholder(tf.float32, [None, 128, 128, 1])
        self.im_dims['EVAL'] = tf.placeholder(tf.int32, [None, 2])

        self.num_images = {'TRAIN': 55000, 'VALID': 5000, 'TEST': 10000}

    def _summaries(self):
        """ Define summaries for TensorBoard """
        tf.summary.scalar("Total_Loss", self.cost)
        tf.summary.scalar("RPN_cls_Loss", self.rpn_cls_loss)
        tf.summary.scalar("RPN_bbox_Loss", self.rpn_bbox_loss)
        tf.summary.scalar("Fast_RCNN_Cls_Loss", self.fast_rcnn_cls_loss)
        tf.summary.scalar("Fast_RCNN_Bbox_Loss", self.fast_rcnn_bbox_loss)
        tf.summary.image("x_train", self.x['TRAIN'])

    def _network(self):
        """ Define the network outputs """
        # Initialize network dicts
        self.cnn = {}
        self.rpn_net = {}
        self.roi_proposal_net = {}
        self.fast_rcnn_net = {}

        # Train network
        with tf.variable_scope('model'):
            self._faster_rcnn(self.x['TRAIN'], self.gt_boxes['TRAIN'], self.im_dims['TRAIN'], 'TRAIN')

        # Eval network => Uses same weights as train network
        with tf.variable_scope('model', reuse=True):
            assert tf.get_variable_scope().reuse is True
            self._faster_rcnn(self.x['EVAL'], None, self.im_dims['EVAL'], 'EVAL')

    def _faster_rcnn(self, x, gt_boxes, im_dims, key):
        # VALID and TEST are both evaluation mode
        eval_mode = True if (key == 'EVAL') else False

        self.cnn[key] = convnet(x, [5, 3, 3, 3, 3], [32, 64, 64, 128, 128], strides=[2, 2, 1, 2, 1])
        feature_maps = self.cnn[key].get_output()
        _feat_stride = self.cnn[key].get_feat_stride()

        # Region Proposal Network (RPN)
        self.rpn_net[key] = rpn(feature_maps, gt_boxes, im_dims, _feat_stride, eval_mode, flags)

        # Roi Pooling
        self.roi_proposal_net[key] = roi_proposal(self.rpn_net[key], gt_boxes, im_dims, eval_mode, flags)

        # R-CNN Classification
        self.fast_rcnn_net[key] = fast_rcnn(feature_maps, self.roi_proposal_net[key], eval_mode)

    def _optimizer(self):
        """ Define losses and initialize optimizer """
        # Losses (come from TRAIN networks)
        self.rpn_cls_loss = self.rpn_net['TRAIN'].get_rpn_cls_loss()
        self.rpn_bbox_loss = self.rpn_net['TRAIN'].get_rpn_bbox_loss()
        self.fast_rcnn_cls_loss = self.fast_rcnn_net['TRAIN'].get_fast_rcnn_cls_loss()
        self.fast_rcnn_bbox_loss = self.fast_rcnn_net['TRAIN'].get_fast_rcnn_bbox_loss()

        # Total Loss (Note: Fast R-CNN bbox refinement loss disabled)
        self.cost = tf.reduce_sum(self.rpn_cls_loss + self.rpn_bbox_loss + self.fast_rcnn_cls_loss)

        # Optimization operation
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)

    def _run_train_iter(self):
        """ Run training iteration"""
        summary, _ = self.sess.run([self.merged, self.optimizer])
        return summary

    def _record_train_metrics(self):
        """ Record training metrics """
        loss, loss1, loss2, loss3 = self.sess.run([self.cost, self.rpn_cls_loss, self.rpn_bbox_loss, self.fast_rcnn_cls_loss])
        self.print_log('Step %d: total loss = %.6f, rpn cls loss = %.6f, rpn bbox loss = %.6f rcnn cls loss = %.6f' %
                       (self.step, loss, loss1, loss2, loss3))

    def train(self):
        """ Run training function. Save model upon completion """
        iterations = int(np.ceil(self.num_images['TRAIN'] / self.flags['batch_size']) * self.flags['num_epochs'])
        self.print_log('Training for %d iterations' % iterations)
        self.step += 1
        for i in tqdm(range(iterations)):
            summary = self._run_train_iter()
            self._record_training_step(summary)
            if self.step % (self.flags['display_step']) == 0:
                self._record_train_metrics()
            if self.step % (self.num_images['TRAIN']) == 0:  # save model every 1 epoch
                self.epochs += 1
                if self.step % (self.num_images['TRAIN'] * 1) == 0:
                    self._save_model(section=self.epochs)
                    self.evaluate(test=False)

    def evaluate(self, test=True):
        """ Evaluate network on the validation set. """
        if test is True:
            key = 'TEST'
            data_directory = flags['data_directory'] + 'Test/'
        else:  # valid
            key = 'VALID'
            data_directory = flags['data_directory'] + 'Valid/'

        print('Detecting images in %s set' % key)
        data_info = (self.num_images[key], flags['num_classes'], flags['classes'])

        tf_inputs = (self.x['EVAL'], self.im_dims['EVAL'])
        tf_outputs = (self.roi_proposal_net['EVAL'].get_rois(),
                      self.fast_rcnn_net['EVAL'].get_cls_prob(),
                      self.fast_rcnn_net['EVAL'].get_bbox_refinement())

        class_metrics = test_net(self.sess, data_directory, data_info, tf_inputs, tf_outputs, vis=self.flags['vis'])
        self.record_eval_metrics(class_metrics, key)

    def record_eval_metrics(self, class_metrics, key):
        """ Record evaluation metrics and print to log and terminal """
        mAP = np.mean(class_metrics)
        self.print_log("Mean Average Precision on " + key + " Set: %f" % mAP)
        fname = self.flags['logging_directory'] + key + '_Accuracy.txt'
        if os.path.isfile(fname):
            self.print_log("Appending to " + key + " file")
            file = open(fname, 'a')
        else:
            self.print_log("Making New " + key + " file")
            file = open(fname, 'w')
        file.write('Epoch: %d' % self.epochs)
        file.write(key + ' set mAP: %f \n' % mAP)
        file.close()

    def close(self):
        Data.exit_threads(self.threads, self.coord)

    def _print_metrics(self):
        self.print_log("Learning Rate: %f" % self.flags['learning_rate'])
        self.print_log("Epochs: %d" % self.flags['num_epochs'])

    @staticmethod
    def read_and_decode(example_serialized):
        """ Read and decode binarized, raw MNIST dataset from .tfrecords file generated by MNIST.py """
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

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Faster RCNN Arguments')
    parser.add_argument('-n', '--run_num', default=0)  # Saves all under /save_directory/model_directory/Model[n]
    parser.add_argument('-e', '--epochs', default=1)  # Number of epochs for which to train the model
    parser.add_argument('-r', '--restore', default=0)  # Binary to restore from a model. 0 = No restore.
    parser.add_argument('-m', '--model_restore', default=1)  # Restores from /save_directory/model_directory/Model[n]
    parser.add_argument('-f', '--file_epoch', default=1)  # Restore filename: 'part_[f].ckpt.meta'
    parser.add_argument('-t', '--train', default=1)  # Binary to train model. 0 = No train.
    parser.add_argument('-v', '--eval', default=1)  # Binary to evalulate model. 0 = No eval.
    parser.add_argument('-y', '--yaml', default='cfgs/clutteredMNIST.yml')  # Configuation Parameter overrides
    parser.add_argument('-l', '--learn_rate', default=0.0001)  # Learning Rate
    parser.add_argument('-i', '--vis', default=0)  # Visualize test results
    parser.add_argument('-g', '--gpu', default=0)  # GPU to use
    args = vars(parser.parse_args())

    # Set Arguments
    flags['run_num'] = int(args['run_num'])
    flags['num_epochs'] = int(args['epochs'])
    if args['restore'] == 0:
        flags['restore'] = False
    else:
        flags['restore'] = True
        flags['restore_file'] = 'part_' + str(args['file_epoch']) + '.ckpt.meta'    
    flags['restore_num'] = int(args['model_restore'])
    flags['file_epoch'] = int(args['file_epoch'])
    if args['yaml'] != 'default':
        dictionary = cfg_from_file('../Models/' + args['yaml'])
        print('Restoring from %s file' % args['yaml'])
    else:
        dictionary = []
        print('Using Default settings')
    flags['learning_rate'] = float(args['learn_rate'])
    flags['vis'] = True if (int(args['vis']) == 1) else False
    flags['gpu'] = int(args['gpu'])
    
    model = FasterRcnnConv5(flags, dictionary)
    if int(args['train']) == 1:
        model.train()
    if int(args['eval']) == 1:
        model.evaluate(test=True)
    model.close()


if __name__ == "__main__":
    main()
