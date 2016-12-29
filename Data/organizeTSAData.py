#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 15:10:30 2016

@author: Kevin Liang

Faster R-CNN expects this file structure:
|-- data/
    |-- Annotations/
         |-- *.txt (Annotation files)
    |-- Images/
         |-- *.png (Image files)
    |-- ImageSets/
         |-- train.txt
These are processing scripts to get the data into this form
"""

import numpy as np
import os
from scipy.misc import imread,imsave

flags = {
    'data_directory': '/media/kd/9ef888dc-a92d-4587-af71-bb562dbc5764/luebeck_2_dataset/',
    'docker_data_directory': '/media/kd/9ef888dc-a92d-4587-af71-bb562dbc5764/justWeapons/',
    #'datasets': ['real_bags','tip_weapons','tip_false_objects'],
    'datasets': ['tip_weapons'],
    'views': 4
    }

def makeAnnotationsDirectory():
    '''
    Creates and populates the Annotations directory expected by Faster RCNN
    '''
    annotations_dir = flags['docker_data_directory'] + "Annotations/"
    
    # Make destination directory
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)
    
    # For each dataset, create an annotations file
    # If no weapon, empty. If weapon, then (classID,upper,left,lower,right)
    for dataset in flags['datasets']:
        print("Converting {0}".format(dataset))
        
        # The summary file containing the IDs of all bags within the dataset
        datafile = flags['data_directory'] + 'description/' + dataset + '.txt'
        with open(datafile) as df:
            # Read in bags one by one from dataset summary file
            for line in df:
                bagID = line.strip()
                # Strip off the .zip if necessary
                if bagID.endswith('.zip'):
                    bagID = bagID[:-4]
                
                for view in range(flags['views']):
                    # Create annotation file for the bag
                    bag_annotation = annotations_dir + bagID + '_' + str(view) + '.txt'
                    with open(bag_annotation, 'w+') as af: 
                        if dataset == 'tip_weapons':
                            left,right,upper,lower = boundWeapon(bagID,view)
                            af.write('1 {0} {1} {2} {3}'.format(upper,left,lower,right))
    
                
def makeImagesDirectory():
    '''
    Creates and populates the Images directory expected by Faster RCNN
    '''
    images_dir = flags['docker_data_directory'] + "Images/"
    
    # Make destination directory
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
    # For each dataset, combine hi and lo images to create 3 channel images for
    # all bags in each dataset
    for dataset in flags['datasets']:
        print("Converting {0}".format(dataset))
        
        # The summary file containing the IDs of all bags within the dataset
        datafile = flags['data_directory'] + 'description/' + dataset + '.txt'
        with open(datafile) as df:
            # Read in bags one by one from dataset summary file
            for line in df:
                bagID = line.strip()
                # Strip off the .zip if necessary
                if bagID.endswith('.zip'):
                    bagID = bagID[:-4]
                
                for view in range(flags['views']):
                    bagImage = convertHiLo2RGB(bagID,view)
                    bagFileName = images_dir + bagID + '_' + str(view) + '.png'
                    imsave(bagFileName,bagImage)
            
            
def makeImageSetsDirectory():
    '''
    Creates and populates the ImageSets directory expected by Faster RCNN
    '''
    imageSets_dir = flags['docker_data_directory'] + "ImageSets/"
    
    # Make destination directory
    if not os.path.exists(imageSets_dir):
        os.makedirs(imageSets_dir)
    
    # Create train.txt and test.txt
    trainfile = imageSets_dir + 'train.txt'
    testfile = imageSets_dir + 'test.txt'
    
    with open(trainfile, 'a') as train, open(testfile, 'a') as test:
        for dataset in flags['datasets']: 
            # tip_weapons dataset already has a train-test split
            if dataset == 'tip_weapons':
                tip_trainfile = flags['data_directory'] + 'description/tip_weapons_train.txt'
                tip_testfile = flags['data_directory'] + 'description/tip_weapons_test.txt'
                with open(tip_trainfile) as tip_train, open(tip_testfile) as tip_test:
                    for line in tip_train:
                        bagID = line.strip()
                        if bagID.endswith('.zip'):
                            bagID = bagID[:-4]
                        for v in range(flags['views']):
                            train.write(bagID + '_' + str(v) + '\n')
                    for line in tip_test:
                        bagID = line.strip()
                        if bagID.endswith('.zip'):
                            bagID = bagID[:-4]
                        for v in range(flags['views']):
                            test.write(bagID + '_' + str(v) + '\n')
            else:                              
                # The summary file containing the IDs of all bags within the dataset
                datafile = flags['data_directory'] + 'description/' + dataset + '.txt'
                with open(datafile) as df:
                    for line in df:
                        # 2:1 split of train:test
                        if np.random.rand()<(2/3):
                            for v in range(flags['views']):
                                train.write(line.strip() + '_' + str(v) + '\n')
                        else:
                            for v in range(flags['views']):
                                test.write(line.strip() + '_' + str(v) + '\n')        
    
    
def boundWeapon(bagID,view):
    '''
    Find the bounding box boundaries of a weapon mask
    
    Args:
        bagID (string): Identifier for the bag found in .txt file in descriptions
        view (int):     Which of the four view of the scanner
        
    Returns:
        left (int): left border coordinate
        right (int): right border coordinate
        upper (int): upper border coordinate
        lower (int): lower border coordinate
    '''
    weapon_file = flags['data_directory'] + 'calibrated_pngs/' + bagID + '_' + str(view) + '_gt.png' 
    weapon = imread(weapon_file)[:,:,0]
    
    # Find the nonzero rows and columns
    rows = np.sum(weapon,axis=0)
    cols = np.sum(weapon,axis=1)
    
    # Bounding box boundaries
    left = np.nonzero(rows)[0][0]
    right = np.nonzero(rows)[0][-1]
    upper = np.nonzero(cols)[0][0]
    lower = np.nonzero(cols)[0][-1]

    return left,right,upper,lower
    

def convertHiLo2RGB(bagID,view):    
    '''
    Bag images are 2 channels (hi,lo). To use weights pre-trained on ImageNet,
    inputs must be 3 channels (RGB). This function combines the hi-lo channels
    and artificially creates a third by taking the mean of the two. Also 
    performs image normalization using transformLogFull
    
    
    Note: not used for weapon masks
    
    Args:
        bagID (string): Identifier for the bag found in .txt file in descriptions
        view (int):     Which of the four view of the scanner
        
    Returns:
        out (numpy array 3D): 3 channel image (hi,lo,mean)
    '''
    filename_hi = flags['data_directory'] + 'calibrated_pngs/' + bagID + '_' + str(view) + '_hi.png'
    filename_lo = flags['data_directory'] + 'calibrated_pngs/' + bagID + '_' + str(view) + '_lo.png'
    
    # Read in the hi and lo energy images and perform normalization
    image_hi = transformLogFull(imread(filename_hi))
    image_lo = transformLogFull(imread(filename_lo))
    
    # Create dummy 3rd channel
    image_mean = (image_hi+image_lo)/2

    # Stack the 3 channels
    out = np.stack((image_lo,image_hi,image_mean),axis=2)
    
    return out
    
    
def transformLogFull(image):
    '''
    Bag images at low and high energies is 16 bits; normalize these using a log
    
    Note: not necessary for weapon masks
    
    Args:
        image (numpy array 2D): image to be normalized
        
    Returns:
        Normalized image
    '''
    maxVal=65536
    maxValInv=1./maxVal
    scaling=65535./np.log(maxValInv)
    
    out = np.minimum(maxVal,np.log((image+1)*maxValInv)*scaling)
    
    return out
        
    
def main():
    print("Assembling data directory structure")
    
    print("Creating Annotations Directory")
    makeAnnotationsDirectory()
    
    print("Creating Images Directory")
    makeImagesDirectory()
    
    print("Creating ImageSets Directory")
    makeImageSetsDirectory()
    
    print ("Finished")

if __name__ == '__main__':
    main()
