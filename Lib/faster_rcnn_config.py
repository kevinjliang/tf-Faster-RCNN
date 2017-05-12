# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 20:47:15 2017

@author: Kevin Liang (Modifications)
"""

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Faster R-CNN config system.

This file specifies default config options for Faster R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Examples of YAML cfg files are located in Models/cfgs.
"""

import os
import os.path as osp
import numpy as np
from distutils import spawn
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from faster_rcnn_config import cfg
cfg = __C


###############################################################################
# Network Architecture
###############################################################################

# Classes: The types of objects the algorithm is trying to find
# First class must be __background__
__C.CLASSES = ['__background__']
__C.NUM_CLASSES = 1

# RPN Anchor Box Scales: Anchor boxes will have dimensions scales*16*ratios in image space
__C.RPN_ANCHOR_SCALES = [8, 16, 32]

# RPN CNN parameters
__C.RPN_OUTPUT_CHANNELS = [512]
__C.RPN_FILTER_SIZES    = [3]

# Fast R-CNN Fully Connected Layer hidden unit number
__C.FRCNN_FC_HIDDEN = [1024, 1024]
# Fast R-CNN dropout keep rate
__C.FRCNN_DROPOUT_KEEP_RATE = 0.5



###############################################################################
# Training options
###############################################################################

__C.TRAIN = edict()

# Learning rate
__C.TRAIN.LEARNING_RATE = 0.001
# Learning rate decay factor
__C.TRAIN.LEARNING_RATE_DECAY = 0.5
# Number of epochs before decaying learning rate 
__C.TRAIN.LEARNING_RATE_DECAY_RATE = 10


# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
#__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
#__C.TRAIN.MAX_SIZE = 1000

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 1 

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE = 128

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.0

# Use horizontally-flipped images during training?
__C.TRAIN.USE_HORZ_FLIPPED = True
# Use vertically-flipped images during training?
__C.TRAIN.USE_VERT_FLIPPED = False

# Train bounding-box refinement in Fast R-CNN
__C.TRAIN.BBOX_REFINE = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
#__C.TRAIN.BBOX_THRESH = 0.5


# Normalize the targets (subtract empirical mean, divide by empirical stddev)
__C.TRAIN.BBOX_NORMALIZE_TARGETS = True
# Deprecated (inside weights)
#__C.TRAIN.BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Normalize the targets using "precomputed" (or made up) means and stdevs
# (BBOX_NORMALIZE_TARGETS must also be True)
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED = False
__C.TRAIN.BBOX_NORMALIZE_MEANS = (0.0, 0.0, 0.0, 0.0)
__C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2)


# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.
#__C.TRAIN.ASPECT_GROUPING = True

# Use RPN to detect objects
#__C.TRAIN.HAS_RPN = True # Default: False
# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# If an anchor satisfied by positive and negative conditions set to negative
__C.TRAIN.RPN_CLOBBER_POSITIVES = False
# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.TRAIN.RPN_BATCHSIZE = 256
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000 
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000 
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TRAIN.RPN_MIN_SIZE = 16
# Deprecated (outside weights)
__C.TRAIN.RPN_BBOX_INSIDE_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# Give the positive RPN examples weight of p * 1 / {num positives}
# and give negatives a weight of (1 - p)
# Set to -1.0 to use uniform example weighting
__C.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# Relative weight of RPN bounding box loss
__C.TRAIN.RPN_BBOX_LAMBDA = 10.0

# Relative weight of Fast RCNN bounding box loss
__C.TRAIN.FRCNN_BBOX_LAMBDA = 1.0



###############################################################################
# Testing options
###############################################################################

__C.TEST = edict()

# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
#__C.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
#__C.TEST.MAX_SIZE = 1000


# Test using bounding-box refinement in Fast R-CNN
# Note: Should not be on if TRAIN.BBOX_REFINE is not also True
__C.TEST.BBOX_REFINE = True

# Propose boxes
__C.TEST.HAS_RPN = True


# NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7 
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 6000 
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 300
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 16

# NMS overlap threshold used post-refinement (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3


# Evaluate with test ground truth (Turn off for deployment, when you don't have gt info)
__C.TEST.GROUNDTRUTH = True
# Plot ground truth boxes on output images. (Turn off if gt boxes are creating too much clutter)
__C.TEST.PLOT_GROUNDTRUTH = True
# Output image colormap (cmap argument to matplotlib.pyplot.imshow())
# Default of 'jet' is standard RGB
__C.TEST.CMAP = 'jet'



###############################################################################
# MISC
###############################################################################

# Relative location of data files
__C.DATA_DIRECTORY = '../Data/'
# Relative location of where of logging directory
__C.SAVE_DIRECTORY = '../Logs/'
# Model directory under logging directory, where 'Model[n]' folder is created
__C.MODEL_DIRECTORY = 'FRCNN/'
# TF Slim restore file for resnet50
__C.RESTORE_SLIM_FILE = '../Data/'

# How much of GPU memory to use (TensorFlow tries to take up entire GPU by default)
__C.VRAM = 0.8

# Image file format ('.png', '.jpg')
__C.IMAGE_FORMAT = '.png'
# Number of bits representing the image
__C.IMAGE_BITDEPTH = 8

# If dataset consists of natural images, subtract pixel means 
# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.NATURAL_IMAGE = True
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# How often to save TensorFlow checkpoint of model parameters (epochs)
__C.CHECKPOINT_RATE = 1 
# How often to evaluate on the validation set (epochs)
__C.VALID_RATE = 1
# How often to show training losses (iterations)
__C.DISPLAY_RATE = 250

# Include objects labeled as "difficult" (PASCAL VOC)
__C.USE_DIFFICULT = False



###############################################################################
###############################################################################

if spawn.find_executable("nvcc"):
    # Use GPU implementation of non-maximum suppression
    __C.USE_GPU_NMS = True

    # Default GPU device id
    __C.GPU_ID = 0
else:
    __C.USE_GPU_NMS = False


def get_output_dir(imdb, weights_filename):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if weights_filename is not None:
        outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value