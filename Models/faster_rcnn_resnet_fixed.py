# -*- coding: utf-8 -*-
"""
Created on Sat Dec 31 13:22:36 2016

@author: Kevin Liang

Faster R-CNN model using ResNet50 as the convolutional feature extractor.
Option to use ImageNet pre-trained weights

Note: We take the feature maps before the last ResNet block
"""

import sys

sys.path.append('../')

from Lib.TensorBase.tensorbase.base import Model
from Lib.faster_rcnn_config import cfg
from Lib.test_aux import test_net
from Lib.train_aux import randomize_training_order, create_feed_dict

from Networks.resnet_v1 import resnetv1
from Networks.faster_rcnn_networks import rpn, roi_proposal, fast_rcnn

from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tqdm import tqdm, trange

import numpy as np
import tensorflow as tf
import argparse
import os

slim = tf.contrib.slim
resnet_arg_scope = resnet_utils.resnet_arg_scope

# Global Dictionary of Flags: Populated in main() with cfg defaults
flags = {}


class FasterRcnnRes50(Model):
    def __init__(self, flags_input, cfg_dict):
        self.epoch = flags_input['FILE_EPOCH'] if flags_input['RESTORE_META'] == 1 else 0
        super().__init__(flags_input, cfg_dict)
        self.flags = self.get_flags()

    def _data(self):
        # Data list for each split
        self.names = {'TRAIN': self._read_names(self.flags['DATA_DIRECTORY'] + 'Names/train.txt'),
                      'VALID': self._read_names(self.flags['DATA_DIRECTORY'] + 'Names/valid.txt'),
                      'TEST': self._read_names(self.flags['DATA_DIRECTORY'] + 'Names/test.txt')}

        # Initialize placeholder dicts
        self.x = {}
        self.gt_boxes = {}
        self.im_dims = {}

        # Train data
        self.x['TRAIN'] = tf.placeholder(tf.float32, [1, None, None, 3], name='image_train')
        self.im_dims['TRAIN'] = tf.placeholder(tf.int32, [None, 2], name='im_dims_train')
        self.gt_boxes['TRAIN'] = tf.placeholder(tf.int32, [None, 5], name='gt_boxes_train')

        # Validation and Test data. No GT Boxes.
        self.x['EVAL'] = tf.placeholder(tf.float32, [1, None, None, 3], name='image_eval')
        self.im_dims['EVAL'] = tf.placeholder(tf.int32, [None, 2], name='im_dims_eval')

    def _network(self):
        """ Define the network outputs """
        # Initialize network dicts
        self.rpn_net = {}
        self.roi_proposal_net = {}
        self.fast_rcnn_net = {}

        # Train network
        with tf.variable_scope("model"):
            with slim.arg_scope(resnet_arg_scope()):
                self._faster_rcnn(self.x['TRAIN'], self.gt_boxes['TRAIN'], self.im_dims['TRAIN'], 'TRAIN')

        with tf.variable_scope("model", reuse=True):
            with slim.arg_scope(resnet_arg_scope()):
                self._faster_rcnn(self.x['EVAL'], None, self.im_dims['EVAL'], 'EVAL')

    def _faster_rcnn(self, x, gt_boxes, im_dims, key):
        # VALID and TEST are both evaluation mode
        eval_mode = True if (key == 'EVAL') else False

        # CNN Feature extractor
        feature_maps = resnetv1(x).build_network()
        # CNN downsampling factor
        _feat_stride = 16

        # Region Proposal Network (RPN)
        self.rpn_net[key] = rpn(feature_maps, gt_boxes, im_dims, _feat_stride, eval_mode)

        # RoI Proposals
        self.roi_proposal_net[key] = roi_proposal(self.rpn_net[key], gt_boxes, im_dims, eval_mode)

        # Fast R-CNN Classification
        self.fast_rcnn_net[key] = fast_rcnn(feature_maps, self.roi_proposal_net[key], eval_mode)

    def _optimizer(self):
        """ Define losses and initialize optimizer """
        # Losses (come from TRAIN networks)
        self.rpn_cls_loss = self.rpn_net['TRAIN'].get_rpn_cls_loss()
        self.rpn_bbox_loss = self.rpn_net['TRAIN'].get_rpn_bbox_loss()
        self.fast_rcnn_cls_loss = self.fast_rcnn_net['TRAIN'].get_fast_rcnn_cls_loss()
        self.fast_rcnn_bbox_loss = self.fast_rcnn_net['TRAIN'].get_fast_rcnn_bbox_loss() * cfg.TRAIN.BBOX_REFINE

        # Total Loss
        self.cost = tf.reduce_sum(
            self.rpn_cls_loss + self.rpn_bbox_loss + self.fast_rcnn_cls_loss + self.fast_rcnn_bbox_loss)

        # Optimizer arguments
        decay_steps = cfg.TRAIN.LEARNING_RATE_DECAY_RATE * len(self.names['TRAIN'])  # Number of Epochs x images/epoch
        learning_rate = tf.train.exponential_decay(learning_rate=cfg.TRAIN.LEARNING_RATE, global_step=self.step,
                                                   decay_steps=decay_steps, decay_rate=cfg.TRAIN.LEARNING_RATE_DECAY,
                                                   staircase=True)

        # Optimizer: ADAM
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.1).minimize(self.cost)

    def _summaries(self):
        """ Define summaries for TensorBoard """
        tf.summary.scalar("Total_Loss", self.cost)
        tf.summary.scalar("RPN_cls_Loss", self.rpn_cls_loss)
        tf.summary.scalar("RPN_bbox_Loss", self.rpn_bbox_loss)
        tf.summary.scalar("Fast_RCNN_Cls_Loss", self.fast_rcnn_cls_loss)
        tf.summary.scalar("Fast_RCNN_Bbox_Loss", self.fast_rcnn_bbox_loss)

    def _run_train_iter(self, feed_dict):
        """ Run training iteration"""
        summary, _ = self.sess.run([self.merged, self.optimizer], feed_dict=feed_dict)
        return summary

    def _record_train_metrics(self, feed_dict):
        """ Run training iteration and record training metrics """
        summary, _, loss, loss1, loss2, loss3, loss4 = self.sess.run([self.merged, self.optimizer, self.cost,
                                                                      self.rpn_cls_loss, self.rpn_bbox_loss,
                                                                      self.fast_rcnn_cls_loss,
                                                                      self.fast_rcnn_bbox_loss],
                                                                     feed_dict=feed_dict)
        self.print_log(
            'Step %d: total loss=%.6f, rpn_cls loss=%.6f, rpn_bbox loss=%.6f, rcnn_cls loss=%.6f, rcnn_bbox loss=%.6f' %
            (self.step, loss, loss1, loss2, loss3, loss4))
        return summary

    def train(self):
        """ Run training function. Save model upon completion """
        self.print_log('Training for %d epochs' % self.flags['NUM_EPOCHS'])

        tf_inputs = (self.x['TRAIN'], self.im_dims['TRAIN'], self.gt_boxes['TRAIN'])

        for self.epoch in trange(1, self.flags['NUM_EPOCHS'] + 1, desc='epochs'):
            train_order = randomize_training_order(len(self.names['TRAIN']))

            for i in tqdm(train_order):
                feed_dict = create_feed_dict(self.flags['DATA_DIRECTORY'], self.names['TRAIN'], tf_inputs, i)

                # Run a training iteration
                if self.step % cfg.DISPLAY_RATE == 0:
                    # Record training metrics every DISPLAY_RATE interval
                    summary = self._record_train_metrics(feed_dict)
                    self._record_training_step(summary)
                else:
                    summary, _ = self.sess.run([self.merged, self.optimizer], feed_dict=feed_dict)
                    self._record_training_step(summary)

                    # Epoch finishe; save model
            if self.epoch % cfg.CHECKPOINT_RATE == 0:
                self._save_model(section=self.epoch)
            # Perform validation
            if self.epoch % cfg.VALID_RATE == 0:
                self.evaluate(test=False)

    def evaluate(self, test=True):
        """ Evaluate network on the validation set. """
        key = 'TEST' if test is True else 'VALID'

        print('Detecting images in %s set' % key)

        tf_inputs = (self.x['EVAL'], self.im_dims['EVAL'])
        tf_outputs = (self.roi_proposal_net['EVAL'].get_rois(),
                      self.fast_rcnn_net['EVAL'].get_cls_prob(),
                      self.fast_rcnn_net['EVAL'].get_bbox_refinement())

        class_metrics = test_net(self.flags['DATA_DIRECTORY'], self.names[key], self.sess, tf_inputs, tf_outputs,
                                 key=key, thresh=0.5, vis=self.flags['VIS'])
        self.record_eval_metrics(class_metrics, key)

    def record_eval_metrics(self, class_metrics, key, display_APs=True):
        """ Record evaluation metrics and print to log and terminal """
        if display_APs:
            for c in range(1, cfg.NUM_CLASSES):
                self.print_log("Average Precision for class {0}: {1:.5}".format(cfg.CLASSES[c], class_metrics[c - 1]))

        mAP = np.mean(class_metrics)
        self.print_log("Mean Average Precision on " + key + " Set: %f" % mAP)

        fname = self.flags['LOGGING_DIRECTORY'] + key + '_Accuracy.txt'
        if os.path.isfile(fname):
            self.print_log("Appending to " + key + " file")
            file = open(fname, 'a')
        else:
            self.print_log("Making New " + key + " file")
            file = open(fname, 'w')
        file.write('Epoch: %d' % self.epoch)
        file.write(key + ' set mAP: %f \n' % mAP)
        file.close()

    def _read_names(self, names_file):
        ''' Read the names.txt file and return a list of all bags '''
        with open(names_file) as f:
            names = f.read().splitlines()
        return names

#    def _set_tf_functions(self):
#        """ Sets up summary writer, saver, and session, with configurable gpu visibility """
#        merged = tf.summary.merge_all()
        #saver = tf.train.Saver()
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.47, visible_device_list=str(self.flags['GPU']))
        #config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
        #sess = tf.Session(config=config)
        #writer = tf.summary.FileWriter(self.flags['LOGGING_DIRECTORY'], sess.graph)
        #return merged, saver, sess, writer


def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Faster R-CNN Networks Arguments')
    parser.add_argument('-n', '--RUN_NUM', default=0,
                        type=int)  # Saves all under /save_directory/model_directory/Model[n]
    parser.add_argument('-e', '--NUM_EPOCHS', default=1, type=int)  # Number of epochs for which to train the model
    parser.add_argument('-r', '--RESTORE_META', default=0, type=int)  # Binary to restore from a model. 0 = No restore.
    parser.add_argument('-m', '--MODEL_RESTORE', default=1,
                        type=int)  # Restores from /save_directory/model_directory/Model[n]
    parser.add_argument('-f', '--FILE_EPOCH', default=1, type=int)  # Restore filename: 'part_[f].ckpt.meta'
    parser.add_argument('-s', '--RESTORE_SLIM', default=1)  # Binary to restore a TF-Slim Model. 0 = No eval.
    parser.add_argument('-t', '--TRAINING', default=1, type=int)  # Binary to train model. 0 = No train.
    parser.add_argument('-v', '--EVAL', default=1, type=int)  # Binary to evalulate model. 0 = No eval.
    parser.add_argument('-y', '--YAML_FILE', default='pascal_voc2007.yml',
                        type=str)  # YAML file to override config defaults
    parser.add_argument('-i', '--VIS', default=0, type=int)  # enable visualizations
    parser.add_argument('-g', '--GPU', default=0, type=int)  # specifiy which GPU to use. Defaults to only one GPU.
    flags = vars(parser.parse_args())

    # Set Arguments
    flags['SEED'] = 1234
    flags['VIS'] = True if (flags['VIS'] == 1) else False
    flags['YAML_FILE'] = 'cfgs/' + flags['YAML_FILE']

    model = FasterRcnnRes50(flags, cfg)
    if flags['TRAINING'] == 1:
        model.train()
    if flags['EVAL'] == 1:
        model.evaluate()


if __name__ == "__main__":
    main()
