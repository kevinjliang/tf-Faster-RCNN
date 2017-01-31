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

from Networks.convnet import convnet
from Networks.faster_rcnn_networks import rpn, roi_proposal, fast_rcnn

import numpy as np
import tensorflow as tf

# Global Dictionary of Flags
flags = {
    'save_directory': './',
    'model_directory': 'conv5/',
    'restore_file': 'part_3.ckpt.meta',
    'restore': True,
    'restore_folder': 2,
    'batch_size': 1,
    'display_step': 50,
    'num_epochs': 10,
    'num_classes': 11,   # 10 digits, +1 for background
    'anchor_scales': [1, 2, 3]
}


class FasterRcnnConv5(Model):
    def __init__(self, flags_input, run_num, restore):
        super().__init__(flags_input, run_num, vram=0.3, restore=restore)
        self.print_log("Seed: %d" % flags['seed'])
        self.threads, self.coord = Data.init_threads(self.sess)

    def _data(self):
        file_train = '/home/dcs41/Documents/tf-Faster-RCNN/Data/data_clutter/clutter_mnist_train.tfrecords'
        self.x, self.gt_boxes, self.im_dims = Data.batch_inputs(self.read_and_decode, file_train,
                                                                batch_size=self.flags['batch_size'])
        file_valid = '/home/dcs41/Documents/tf-Faster-RCNN/Data/data_clutter/clutter_mnist_valid.tfrecords'
        self.x_valid, self.gt_boxes_valid, self.im_dims_valid = Data.batch_inputs(self.read_and_decode, file_valid,
                                                                                  mode="eval", batch_size=1, num_threads=1, num_readers=1)
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
        tf.summary.image("x", self.x)

    def _network(self):
        ''' Define the network outputs '''
        with tf.variable_scope('model'):
            self.cnn = convnet(self.x, [5, 3, 3, 3, 3], [64, 96, 128, 172, 256], strides=[2, 2, 2, 2, 2])
            featureMaps = self.cnn.get_output()
            _feat_stride = self.cnn.get_feat_stride()

            # Region Proposal Network (RPN)
            self.rpn_net = rpn(featureMaps, self.gt_boxes, self.im_dims, _feat_stride, flags)

            rpn_cls_score = self.rpn_net.get_rpn_cls_score()
            rpn_bbox_pred = self.rpn_net.get_rpn_bbox_pred()

            # Roi Pooling
            roi_proposal_net = roi_proposal(rpn_cls_score, rpn_bbox_pred, self.gt_boxes, self.im_dims, flags)

            # R-CNN Classification
            self.fast_rcnn_net = fast_rcnn(featureMaps, roi_proposal_net)

        with tf.variable_scope('model', reuse=True):
            assert tf.get_variable_scope().reuse is True
            self.cnn_valid = convnet(self.x_valid, [5, 3, 3, 3, 3], [64, 96, 128, 172, 256], strides=[2, 2, 2, 2, 2])
            featureMaps_valid = self.cnn_valid.get_output()
            _feat_stride_valid = self.cnn_valid.get_feat_stride()

            # Region Proposal Network (RPN)
            self.rpn_net_valid = rpn(featureMaps_valid, self.gt_boxes_valid, self.im_dims_valid, _feat_stride_valid, flags)

            rpn_cls_score_valid = self.rpn_net_valid.get_rpn_cls_score()
            rpn_bbox_pred_valid = self.rpn_net_valid.get_rpn_bbox_pred()

            # Roi Pooling
            self.roi_proposal_net_valid = roi_proposal(rpn_cls_score_valid, rpn_bbox_pred_valid, self.gt_boxes_valid, self.im_dims_valid, flags)

            # R-CNN Classification
            self.fast_rcnn_net_valid = fast_rcnn(featureMaps_valid, self.roi_proposal_net_valid)

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
            x, gt_boxes = self.sess.run([self.x, self.gt_boxes])
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
        for i in range(3):
            summary = self._run_train_iter()
            if self.step % self.flags['display_step'] == 0:
                self._record_train_metrics()
                bbox, cls = self.sess.run([self.fast_rcnn_net.get_bbox_refinement(), self.fast_rcnn_net.get_cls_score()])
                self.print_log('Number of predictions: %d' % bbox.shape[0])
            if (self.step % 2) == 0:
                self._save_model(section=epochs)
                epochs += 1
            self._record_training_step(summary)
            print(self.step)
        Data.exit_threads(self.threads, self.coord)  # Exit Queues

    def eval(self):
        """ Loop Through Evaluation Dataset and calculate metrics """
        info = dict()
        for c in range(flags['num_classes']):
            info[c] = list()
        for i in range(self.num_valid_images):
            bboxes, cls_score, gt_boxes, image = self.sess.run([self.roi_proposal_net_valid.get_rois(),
                                                       self.fast_rcnn_net_valid.get_cls_score(), self.gt_boxes_valid,
                                                       self.x_valid])
            print(gt_boxes)
            self.vis_detections(np.squeeze(image[0]), str(gt_boxes[0][4]), gt_boxes, bboxes)
        Data.exit_threads(self.threads, self.coord)  # Exit Queues

    def vis_detections(self, im, class_name, gt_boxes, dets):
        """Visual debugging of detections."""
        import matplotlib
        matplotlib.use('TkAgg')  # For Mac OS
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        fig, ax = plt.subplots(1)
        for i in range(np.minimum(10, dets.shape[0])):
            bbox = dets[i,1:]
            print(bbox)
            ax.imshow(np.squeeze(im), cmap="gray")
            self.plot_patch(ax, patches, bbox, gt=False)
        plt.title(class_name)
        self.plot_patch(ax, patches, gt_boxes[0][:4], gt=True)

        # Display Final composite image
        plt.show()

    @staticmethod
    def plot_patch(ax, patches, bbox, gt):
        if gt is True:
            color = 'g'
        else:
            color = 'r'
        # Calculate Bounding Box Rectangle and plot it
        width = bbox[3] - bbox[1]
        height = bbox[2] - bbox[0]
        rect = patches.Rectangle((bbox[1], bbox[0]), height, width, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)

    @staticmethod
    def read_and_decode(example_serialized):
        """ Read and decode binarized, raw MNIST dataset from .tfrecords file generated by clutterMNIST.py """
        features = tf.parse_single_example(
            example_serialized,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'gt_boxes': tf.FixedLenFeature([5], tf.int64, default_value=[-1] * 5),
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
    run_num = sys.argv[1]
    model = FasterRcnnConv5(flags, run_num=run_num, restore=flags['restore_folder'])
    model.eval()


if __name__ == "__main__":
    main()