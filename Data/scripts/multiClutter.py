#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:07:25 2017

@author: Kevin Liang

Generate data for HRI project

Test: 1202 images

Modules:
    convert_data_tfrecords()
    aux_convert_tfrecords()
    write()
"""

from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
from scipy.ndimage.interpolation import zoom

import numpy as np
import os
import shutil
import tensorflow as tf

# Global Flag Dictionary
flags = {
    'nums': {"train": 55000, "valid": 5000, "test": 602},
    'num_classes': 10,
    'im_dims': 128,
    'num_digits': 3,
    'A': np.array([0, 16,  28,  93, 119, 166, 185, 233, 238, 284, 295]),       # Randomly generated
    'B': 301+np.array([0, 27,  31,  99, 112, 141, 165, 207, 215, 251, 297])    # Randomly generated
}

def main():
    """Downloads and Converts MNIST dataset to three .tfrecords files (train, valid, test)
    Takes variable number of labels."""
    
    # Load and Convert Data
    all_data, all_labels = load_data()
    shutil.rmtree("MNIST_data")

    directory = '../HRI_clutteredMNIST/'
    make_directory(directory)

    convert_train(all_data[0], all_labels[0], directory)
    convert_valid(all_data[1], all_labels[1], directory+'Valid/')
    convert_test(all_data[2], all_labels[2], directory+'Test/')
    

def convert_train(data, labels, data_directory):
    print('Processing Train Data')
    tf_writer = tf.python_io.TFRecordWriter(data_directory + 'hri_mnist.tfrecords')
    
    for i in tqdm(range(flags['nums']['train'])):
        im, gt_boxes = gen_nCluttered(data, labels, flags['im_dims'], flags['num_digits'])
        
        im = np.float32(im.flatten()).tostring()
        gt_boxes = np.int32(np.array(gt_boxes).flatten()).tostring()
        
        write(im, gt_boxes, [flags['im_dims'],flags['im_dims']], tf_writer)
        
        
def convert_valid(data, labels, data_directory):
    print('Processing Valid Data')
    make_directory(data_directory + 'Images/')
    make_directory(data_directory + 'Annotations/')
    make_directory(data_directory + 'Names')
    
    for i in tqdm(range(flags['nums']['valid'])):
        im, gt_boxes = gen_nCluttered(data, labels, flags['im_dims'], flags['num_digits'])
        
        fname = 'img' + str(i)
        
        # Save PNG, Annotation
        np.save(data_directory + 'Images/' + fname, np.float32(im))
        np.savetxt(data_directory + 'Annotations/' + fname + '.txt', np.array(gt_boxes), fmt='%i')
        with open(data_directory + 'Names/names.txt', 'a') as f:
            f.write(fname + '\n')
            
def convert_test(data, labels, data_directory):
    print('Processing Test Data')
    make_directory(data_directory + 'Images/')
    make_directory(data_directory + 'Annotations/')
    make_directory(data_directory + 'Names')
    
    # Indices of the odd and even MNIST digits
#    even_idx = np.where(np.equal(np.mod(labels, 2),0))
    odd_idx = np.where(np.mod(labels, 2))
    
    for i in tqdm(range(flags['nums']['test'])):
        if i in flags['A'] or i in flags['B']:
            # Generate cluttered MNIST example with a single even number
            while(True):
                im, gt_boxes = gen_nCluttered(data, labels, flags['im_dims'], flags['num_digits'])
                
                gen_labels = np.array(gt_boxes)[:,4]
                gen_evens = np.equal(np.mod(gen_labels, 2),0)
                
                if sum(gen_evens) == 1:
                    break
        else:
            # Generate cluttered MNIST example with only odds
            im, gt_boxes = gen_nCluttered(data[odd_idx], labels[odd_idx], flags['im_dims'], flags['num_digits'])
            
        fname = 'img' + str(i)
        
        # Save PNG, Annotation
        np.save(data_directory + 'Images/' + fname, np.float32(im))
        np.savetxt(data_directory + 'Annotations/' + fname + '.txt', np.array(gt_boxes), fmt='%i')
        with open(data_directory + 'Names/names.txt', 'a') as f:
            f.write(fname + '\n')
        

def gen_nCluttered(data, labels, im_dims, num_digits):
    # Initialize Blank image_out
    image_out = np.zeros([im_dims, im_dims])
    max_val = 0
    gt_boxes = list()
    
    for i in range(num_digits):
        # Choose digit
        idx = np.random.randint(len(labels))
        digit = data[idx,:].reshape((28,28))
        label = labels[idx]
        
        # Randomly Scale image
        h = np.random.randint(low=int(28/1.5), high=int(28*1.5))
        w = np.random.randint(low=int(28/1.5), high=int(28*1.5))
        digit = zoom(digit, (h/28, w/28))
        
        while(True):
            # Randomly choose location in image_out
            x = np.random.randint(low=0, high=im_dims - w)
            y = np.random.randint(low=0, high=im_dims - h)
        
            # Ensure that digit doesn't overlap with another
            if np.sum(image_out[y:y + h, x:x + w])==0:
                break
            
        # Insert digit into blank full size image and get max
        embedded = np.zeros([im_dims, im_dims])
        embedded[y:y + h, x:x + w] += digit
        max_val = max(embedded.max(), max_val)
        
        # Tighten box
        rows = np.sum(embedded, axis=0).round(1)
        cols = np.sum(embedded, axis=1).round(1)

        left = np.nonzero(rows)[0][0]
        right = np.nonzero(rows)[0][-1]
        upper = np.nonzero(cols)[0][0]
        lower = np.nonzero(cols)[0][-1]
        
        # If box is too narrow or too short, pad it out to >12
        width = right - left
        if width < 12:
            pad = np.ceil((12 - width)/2)
            left  = int(left - pad)
            right = int(right + pad)

        height = lower - upper
        if height < 12:
            pad = np.ceil((12 - height)/2)
            upper = int(upper - pad)
            lower = int(lower + pad)
            
        # Save Ground Truth Bounding boxes with Label in 4th position
        if labels[idx] == 0:  # Faster RCNN regards 0 as background, so change the label for all zeros to 10
            label = 10
        gt_box = [int(left), int(upper), int(right), int(lower), int(label)]
        
        # Save digit insertion
        image_out = image_out + embedded
        gt_boxes.append(gt_box)
    
    # Add in clutter patches
    for j in range(int(im_dims)):
        # Randomly select MNIST digit
        idx = np.random.randint(len(labels))
        fragment = data[idx,:].reshape(28,28)
        
        # Randomly select patch of selected digit
        px = np.random.randint(low=0, high=28 - 8)
        py = np.random.randint(low=0, high=28 - 8)

        # Randomly choose location to insert clutter
        x = np.random.randint(low=0, high=im_dims - 8)
        y = np.random.randint(low=0, high=im_dims - 8)
        
        # Insert digit fragment, but not on top of digits
        if np.sum(image_out[y:(y+8), x:(x+8)]) == 0:
            image_out[y:(y+8), x:(x+8)] += fragment[py:(py+8), px:(px+8)]
    
    # Normalize any over-saturated pixels
    image_out = np.clip(image_out, 0, max_val)
        
    # Subtract mean from image and scale to be between -1 and 1
    image_out -= image_out.mean()
    image_out = image_out / np.abs(image_out).max()

    return image_out, gt_boxes                   

                    
def write(pixels, gt_boxes, dims, writer):
    """Write image pixels and label from one example to .tfrecords file"""
    example = tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
          # Features contains a map of string to Feature proto objects
          feature={
            # A Feature contains one of either a int64_list,
            # float_list, or bytes_list
            'dims': _int64_list_features(dims),
            'gt_boxes': _bytes_features(gt_boxes),
            'image': _bytes_features(pixels)
    }))
    # Use the proto object to serialize the example to a string and write to disk
    serialized = example.SerializeToString()
    writer.write(serialized)
    

def load_data():
    """ Download MNIST data from TensorFlow package, """
    mnist = input_data.read_data_sets("MNIST_data", one_hot=False)
    train_data = mnist.train.images
    test_data = mnist.test.images
    valid_data = mnist.validation.images
    train_label = mnist.train.labels
    test_label = mnist.test.labels
    valid_label = mnist.validation.labels
    all_data = [train_data, valid_data, test_data]
    all_labels = [train_label, valid_label, test_label]
    return all_data, all_labels


def make_directory(folder_path):
    """Creates directory if it does not exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def _int64_features(value):
    """Value takes a the form of a single integer"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_list_features(list_ints):
    """Value takes a the form of a list of integers"""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_ints))


def _bytes_features(value):
    """Value takes the form of a string of bytes data"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == "__main__":
    main()