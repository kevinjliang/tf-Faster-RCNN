#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 15:08:33 2017

@author: Kevin Liang 
"""

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------


import numpy as np
import scipy
import os
import Pickle
from imdb import imdb

class clutteredMNIST(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'clutteredMNIST')
        self._image_set = image_set
        self._data_path = './clutteredMNIST'
        self._classes = ('__background__', # always index 0
                         '0','1','2','3','4','5','6','7','8','9')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb
        
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)
                
    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])
    
    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'Images',
                                  index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path   
        
    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index
        
    
    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = Pickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_clutteredMNIST_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            Pickle.dump(gt_roidb, fid)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb
        
    def _load_clutteredMNIST_annotation(self, index):
        """
        Load image and bounding boxes info from txt file.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.txt')
        objs = np.loadtxt(filename)
        num_objs = objs.shape[0]

        boxes = objs[:,:4]
        gt_classes = self._class_to_ind[objs[:4]]
        overlaps = np.zeros((num_objs,self.num_classes),dtype=np.float32)
        overlaps[np.arange(num_objs),gt_classes] = 1.0
        overlaps = scipy.sparse.csr_matrix(overlaps)        
        seg_areas = (boxes[:,2]-boxes[:,0] + 1) * (boxes[:,3]-boxes[:,1] + 1)
        
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}