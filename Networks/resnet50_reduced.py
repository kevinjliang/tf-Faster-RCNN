# -*- coding: utf-8 -*-
"""
Created on Thurs Feb 9 10:55:07 2017

@author: Dan Salo

ResNet-50 feature extractor from TF-Slim

Last bottleneck group block has been removed 
"""

import sys
sys.path.append('../')

import tensorflow as tf

from tensorflow.contrib.slim.python.slim.nets import resnet_utils

slim = tf.contrib.slim
resnet_arg_scope = resnet_utils.resnet_arg_scope


@slim.add_arg_scope
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1, outputs_collections=None, scope=None):
    with tf.variable_scope(scope, 'bottleneck_v1', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(inputs, depth, [1, 1], stride=stride, activation_fn=None, scope='shortcut')
        residual = slim.conv2d(inputs, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1, activation_fn=None, scope='conv3')
        output = tf.nn.relu(shortcut + residual)

    return slim.utils.collect_named_outputs(outputs_collections, sc.original_name_scope, output)

def resnet_v1_block(scope, base_depth, num_units, stride):
    """Helper function for creating a resnet_v1 bottleneck block.
    Args:
        scope: The scope of the block.
        base_depth: The depth of the bottleneck layer for each unit.
        num_units: The number of units in the block.
        stride: The stride of the block, implemented as a stride in the last unit.
        All other units have stride=1.
        Returns:
            A resnet_v1 bottleneck block.
    """
    return resnet_utils.Block(scope, bottleneck, [{
            'depth': base_depth * 4,
            'depth_bottleneck': base_depth,
            'stride': 1
            }] * (num_units - 1) + [{
                    'depth': base_depth * 4,
                    'depth_bottleneck': base_depth,
                    'stride': stride
                    }])

def resnet50_reduced(inputs, is_training=True, output_stride=None, include_root_block=True, reuse=None, scope=None):

    # These are the blocks for resnet 50
    blocks = [
        resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
        resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
#        resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
        ]

    # Initialize Model
    with tf.variable_scope(scope, 'resnet_v1_50', [inputs], reuse=reuse):
        with slim.arg_scope([slim.conv2d, bottleneck, resnet_utils.stack_blocks_dense]):
            with slim.arg_scope([slim.batch_norm], is_training=is_training) as scope:
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError('The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
    return net
