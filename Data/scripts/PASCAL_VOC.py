#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:35:40 2017

@author: Kevin Liang (Modifications)
"""

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import sys
sys.path.append('../../')

from Lib.faster_rcnn_config import cfg, cfg_from_file

import argparse
import numpy as np
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='PASCAL_VOC Arguments')
    parser.add_argument('-n', '--year', default='2007')
    parser.add_argument('-y', '--yaml', default='pascal_voc2007.yml')  # YAML file to override config defaults
    args = vars(parser.parse_args())
    
    if args['yaml'] != 'default': 
        dictionary = cfg_from_file('../../Models/cfgs/' + args['yaml'])
        print('Restoring from %s file' % args['yaml'])
    else:
        dictionary = []
        print('Using Default settings')
    
    gen_Annotations_dir(args['year'])
    gen_Images_dir(args['year'])
    gen_Names_dir(args['year'])
    
    
def gen_Annotations_dir(year):
    """
    For each image, generate a corresponding annotations file
    Each line represents an object/bounding box: (x1,y1,x2,y2,label)
    
    label is an integer index, corresponding to the indices of the CLASSES entry
    of pascal_voc2007.yml (0 corresponds to '__background__')
    """
    annotations_dir = '../' + cfg.DATA_DIRECTORY + 'Annotations/'
    make_directory(annotations_dir)
    
    src_dir = '../' + cfg.DATA_DIRECTORY + '../VOCdevkit' + year + '/VOC' + year + '/Annotations/'
    
    class2ind = dict(zip(cfg.CLASSES, range(cfg.NUM_CLASSES)))
    
    # Load all annotations and convert to the appropriate form
    for xml_file in tqdm(os.listdir(src_dir)):
        if os.path.isfile(src_dir + xml_file):
            boxes = _load_pascal_annotation(src_dir + xml_file, class2ind)
            fileID = xml_file[:-4]
            np.savetxt(annotations_dir + fileID + '.txt', np.array(boxes), fmt='%i')
            
def gen_Images_dir(year):
    """ Just create soft link to devkit, to save space """
    images_dir = '../' + cfg.DATA_DIRECTORY + 'Images'
    src_dir = '../VOCdevkit' + year + '/VOC' + year + '/JPEGImages/'
    
    os.symlink(src_dir, images_dir)
            
            
def gen_Names_dir(year):
    """ 
    Combine original training and validation sets of PASCAL VOC into training
    
    Split original test into new validation and test sets (approx 1:1)
    """    
    names_dir = '../' + cfg.DATA_DIRECTORY + 'Names/'
    make_directory(names_dir)
    
    src_dir = '../' + cfg.DATA_DIRECTORY + '../VOCdevkit' + year + '/VOC' + year + '/ImageSets/Main/'
    
    ## Convert Train list (combine train and valid)
    train_src = src_dir + 'trainval.txt'
    train_dest = names_dir + 'train.txt'
    
    # Clobber the old names file
    delete_file(train_dest)
    
    with open(train_src) as s, open(train_dest, 'a') as d:
        for line in tqdm(s):
            s = line# [:-4] # Strip off the ' -1\n'
            d.write(s)
            
            
    ## Convert Test (split test into test and valid)
    test_src = src_dir + 'test.txt'
    
    valid_dest = names_dir + 'valid.txt'
    test_dest = names_dir + 'test.txt'
    
    # Clobber the old names file
    delete_file(valid_dest)
    delete_file(test_dest)
    
    with open(test_src) as s, open(valid_dest, 'a') as vd, open(test_dest, 'a') as td:
        for line in tqdm(s):
            s = line#[:-4] # Strip off the ' -1\n'
            
            # Randomly assign to valid and test according to split:
            if np.random.binomial(1, 0.5):
                vd.write(s)
            else:
                td.write(s)

                
def _load_pascal_annotation(filename, class2ind):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    tree = ET.parse(filename)
    objs = tree.findall('object')
    
    if not cfg.USE_DIFFICULT:
        # Exclude the samples labeled as difficult
        non_diff_objs = [
            obj for obj in objs if int(obj.find('difficult').text) == 0]
        objs = non_diff_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 5), dtype=np.uint16)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1
        cls = class2ind[obj.find('name').text.lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2, cls]

    return boxes
            
            
###############################################################################
# Miscellaneous
###############################################################################     
def make_directory(folder_path):
    """Creates directory if it does not exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
def delete_file(filename):
    try:
        os.remove(filename)
    except OSError:
        pass
        
def make_Im_An_Na_directories(data_directory):
    '''Creates the Images-Annotations-Names directories for a data split'''
    make_directory(data_directory + 'Images/')
    make_directory(data_directory + 'Annotations/')
    make_directory(data_directory + 'Names/')


            
if __name__ == "__main__":
    main()
