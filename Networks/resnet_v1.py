# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He and Xinlei Chen
# --------------------------------------------------------

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1

from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from Lib.faster_rcnn_config import cfg


def resnet_arg_scope(trainable=True,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        # NOTE 'is_training' here does not work because inside resnet it gets reset:
        # https://github.com/tensorflow/models/blob/master/slim/nets/resnet_v1.py#L187
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': cfg.RESNET.BN_TRAIN,
        'updates_collections': ops.GraphKeys.UPDATE_OPS
    }

    with arg_scope(
            [slim.conv2d],
            weights_regularizer=regularizers.l2_regularizer(weight_decay),
            weights_initializer=initializers.variance_scaling_initializer(),
            trainable=trainable,
            activation_fn=nn_ops.relu,
            normalizer_fn=layers.batch_norm,
            normalizer_params=batch_norm_params):
        with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

class resnetv1:
    def __init__(self, x, num_layers=50):
        self._num_layers = num_layers
        self._image = x
        self._arch = 'res_v1_%d' % num_layers
        self._resnet_scope = 'resnet_v1_%d' % num_layers

    # Do the first few layers manually, because 'SAME' padding can behave inconsistently
    # for images of different sizes: sometimes 0, sometimes 1
    def build_base(self):
        with tf.variable_scope(self._resnet_scope, self._resnet_scope):
            net = resnet_utils.conv2d_same(self._image, 64, 7, stride=2, scope='conv1')
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')
        return net

    def build_network(self):
        # select initializers

        bottleneck = resnet_v1.bottleneck
        blocks = [resnet_utils.Block('block1', bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
                  resnet_utils.Block('block2', bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
                  resnet_utils.Block('block3', bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 1)]),
                  resnet_utils.Block('block4', bottleneck, [(2048, 512, 1)] * 3)]

        assert (0 <= cfg.RESNET.FIXED_BLOCKS < 4)
        if cfg.RESNET.FIXED_BLOCKS == 3:
            with slim.arg_scope(resnet_arg_scope(trainable=False)):
                net = self.build_base()
                net_conv4, _ = resnet_v1.resnet_v1(net,
                                                    blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                                    global_pool=False,
                                                    include_root_block=False,
                                                    scope=self._resnet_scope)
        elif cfg.RESNET.FIXED_BLOCKS > 0:
            with slim.arg_scope(resnet_arg_scope(trainable=False)):
                net = self.build_base()
                net, _ = resnet_v1.resnet_v1(net,
                                             blocks[0:cfg.RESNET.FIXED_BLOCKS],
                                             global_pool=False,
                                             include_root_block=False,
                                             scope=self._resnet_scope)

            with slim.arg_scope(resnet_arg_scope()):
                net_conv4, _ = resnet_v1.resnet_v1(net,
                                                    blocks[cfg.RESNET.FIXED_BLOCKS:-1],
                                                    global_pool=False,
                                                    include_root_block=False,
                                                    scope=self._resnet_scope)
        else:    # cfg.RESNET.FIXED_BLOCKS == 0
            with slim.arg_scope(resnet_arg_scope()):
                net = self.build_base()
                net_conv4, _ = resnet_v1.resnet_v1(net,
                                                    blocks[0:-1],
                                                    global_pool=False,
                                                    include_root_block=False,
                                                    scope=self._resnet_scope)
        return net_conv4
