#!/usr/bin/env python

"""
@author: Dan Salo, Jan 2017
Original code by @kevinjliang

Purpose: Create partially-labeled MNIST dataset in .tfrecords format. Number of labels specified by user.

Modules:
    convert_data_tfrecords()
    aux_convert_tfrecords()
    write()
"""

from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
from scipy.ndimage.interpolation import zoom

import argparse
import numpy as np
import os
import random
import shutil
import tensorflow as tf


# Global Flag Dictionary
flags = {
    'nums': {"train": 55000, "valid": 5000, "test": 10000},
    'all_names': ["train", "valid", "test"],
    'num_classes': 10,
}


def main():
    """Downloads and Converts MNIST dataset to three .tfrecords files (train, valid, test)
    Takes variable number of labels."""

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Clutter MNIST Arguments')
    parser.add_argument('-i', '--dim', default=128)
    parser.add_argument('-t', '--test', default="PNG")
    parser.add_argument('-c', '--clutter', default=1)
    parser.add_argument('-s', '--semi', default=0)
    parser.add_argument('-l', '--label_list', default=[10, 50, 100, 500])
    args = vars(parser.parse_args())

    # Load and Convert Data
    all_data, all_labels = load_data()
    shutil.rmtree("MNIST_data")
    all_names = flags['all_names']

    if int(args['clutter']) == 1:
        gen_fn = generate_cluttered_digit
        prefix = 'clutter'
        print('Generating cluttered MNIST')
    else:  # translated MNIST
        gen_fn = generate_translated_digit
        prefix = 'trans'
        print('Generating translated MNIST')
    directory = '../data_' + prefix + '/'
    make_directory(directory)

    if int(args['semi']) == 0:
        label_list = [55000]
    else:
        label_list = args['label_list']

    if args['test'] != 'PNG':
        print('Saving Test Set as TFRecords file.')
        convert_tfrecords(all_data, all_labels, all_names, gen_fn, args['dim'], directory, label_list, prefix)
    else:
        print('Saving Test Set as PNG and Text files.')
        convert_test_and_valid_png(all_data[1:], all_labels[1:], all_names[1:], gen_fn, args['dim'], directory)
        convert_tfrecords([all_data[0]], [all_labels[0]], [all_names[0]], gen_fn, args['dim'], directory, label_list, prefix)


def convert_tfrecords(all_data, all_labels, all_names, gen_fn, image_dim, data_directory, label_list, prefix):
    """ Saves MNIST images and labels in .tfrecords format with one train file
    This function is not used in most of our semi-supervised models as we want to have balance labels in minibatches

    :param all_data: list of train, test, and validation pre-loaded images
    :param all_labels: list of train, test, and validation pre-loaded labels
    :param all_names: list of "train" or "train", "valid", and "test"
    :param image_dim: integer, side length of square output image
    :param data_directory: string of where .tfrecords files will be saved
    """

    for l in range(len(label_list)):    # Loop through all number of labels
        for d in range(len(all_data)):  # Loop through [train, valid, test] for all number of labeled images

            # Initialize
            num_labels = label_list[l]
            data = all_data[d]
            labels = all_labels[d]
            name = all_names[d]

            # Create writers
            if name == 'train' and num_labels != flags['nums'][name]:
                writer_labeled = tf.python_io.TFRecordWriter(data_directory + prefix + '_mnist_' + str(num_labels) + "_" +
                                                             name + "_labeled.tfrecords")
                writer_unlabeled = tf.python_io.TFRecordWriter(data_directory + prefix + '_mnist_' + str(num_labels) + "_" +
                                                               name + "_unlabeled.tfrecords")
                examples_labeled = list()
                examples_unlabeled = list()
                num_samples = np.zeros(flags['num_classes'])
            else:
                writer = tf.python_io.TFRecordWriter(data_directory + prefix + "_mnist_" + name + ".tfrecords")
                examples = list()

            # Iterate over each example and append to list
            for example_idx in tqdm(range(flags['nums'][name])):
                label_np = labels[example_idx].astype("int32")
                label = label_np.tolist()
                pixels, gt_box = gen_fn(data[example_idx].reshape((28, 28)), image_dim, label, data)
                pixels = np.float32(pixels.flatten())

                # Write example to file via writer object
                if name == "train" and num_labels != flags['nums'][name]:
                    if num_samples[label_np] < num_labels:
                        num_samples[label_np] += 1
                        examples_labeled.append((pixels.tostring(), gt_box, [image_dim, image_dim]))
                        print(num_samples)
                    else:
                        examples_unlabeled.append((pixels.tostring(), gt_box, [image_dim, image_dim]))
                else:
                    examples.append((pixels.tostring(), gt_box, [image_dim, image_dim]))

            # Shuffle all examples. This is imperative for good mixing with TF queueing and shuffling.
            if name == "train" and num_labels != flags['nums'][name]:
                random.shuffle(examples_labeled)
                random.shuffle(examples_unlabeled)
                for idx_labeled in tqdm(range(len(examples_labeled))):
                    write(examples_labeled[idx_labeled][0], examples_labeled[idx_labeled][1],
                          examples_labeled[idx_labeled][2], writer_labeled)
                for idx_unlabeled in tqdm(range(len(examples_unlabeled))):
                    write(examples_unlabeled[idx_unlabeled][0], examples_unlabeled[idx_unlabeled][1],
                          examples_unlabeled[idx_unlabeled][2], writer_unlabeled)
            else:
                random.shuffle(examples)
                for idx in tqdm(range(len(examples))):
                    write(examples[idx][0], examples[idx][1], examples[idx][2], writer)


def convert_test_and_valid_png(all_data, all_labels, all_names, gen_fn, image_dim, data_directory):
    """
    Converts Test Dataset into PNGs and saves them in a folder.
    Annotations are saved in another folder in the same directory. All base filenames are saved into yet another folder,
    again in the same directory.
    :param all_data: list of test/valid images
    :param all_labels: list of test/valid labels
    :param all_names: list of "valid" and "test"
    :param image_dim: int, side of square image
    :param data_directory: string, location of folders to be saved
    """

    base = data_directory
    # Loop through [train, valid, test] for all number of labeled images
    for d in range(len(all_data)):

        # Initialize
        data = all_data[d]
        labels = all_labels[d]
        name = all_names[d]
        if name == "valid":
            print('Processing Valid Data')
            directory = base + 'Valid/'
        else:  # name == "test
            print('Processing Test Data')
            directory = base + 'Test/'

        # Make the 3 folders if they don't exist already
        make_directory(directory + 'Images/')
        make_directory(directory + 'Annotations/')
        make_directory(directory + 'Names')

        filenames = list()

        # Loop through all data and save pngs in Images folder, text files in Annotations, a single text file in Names
        for example_idx in tqdm(range(flags['nums'][name])):

            # Generate pixels and ground truth bounding box and label
            label_np = labels[example_idx].astype("int32")
            label = label_np.tolist()
            pixels, gt_box = gen_fn(data[example_idx].reshape((28, 28)), image_dim, label, data)
            fname = 'img' + str(example_idx)

            # Save PNG, Annotation
            np.save(directory + 'Images/' + fname, np.float32(pixels))
            np.savetxt(directory + 'Annotations/' + fname + '.txt', np.array([gt_box]), fmt='%i')
            filenames.append(fname)

        # Save List of Base Filenames in Names folder
        names = open(directory + 'Names/names.txt', 'w')
        for fname in filenames:
            names.write("%s\n" % fname)


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
            'gt_boxes': _int64_list_features(gt_boxes),
            'image': _bytes_features(pixels)
    }))
    # Use the proto object to serialize the example to a string and write to disk
    serialized = example.SerializeToString()
    writer.write(serialized)


def generate_translated_digit(input_image, image_dim, label, data):
    """
    :param input_image: input image to be stored
    :param image_dim: int, side length of square image
    :param data: entire dataset in memory from which we will pick fragments
    :param label: single integer representing the digit in the cluttered image
    :return: image_out: ndarray size image_dim x image_dim with digit randomly placed with
    8 x 8 patches of images with input noise.
    """

    # Initialize Blank image_out
    image_out = np.zeros([image_dim, image_dim])

    # Randomly Scale image
    h = np.random.randint(low=int(28/2), high=int(28*2))
    w = np.random.randint(low=int(28/2), high=int(28*2))
    digit = zoom(input_image, (h/28, w/28))

    # Randomly choose location in image_out and save in bbox list
    x = np.random.randint(low=0, high=image_dim - w)
    y = np.random.randint(low=0, high=image_dim - h)

    # Insert digit into image_out and get max
    image_out[y:y + h, x:x + w] += digit
    max_val = image_out.max()

    # Save Ground Truth Bounding boxes with Label in 4th position
    if label == 0:  # Faster RCNN regards 0 as background, so change the label for all zeros to 10
        label = 10
    gt_box = [x, y, x+w, y+h, label]

    # Don't add any clutter here. Only black space.

    return image_out, gt_box


def generate_cluttered_digit(input_image, image_dim, label, data):
    """
    :param input_image: input image to be stored
    :param image_dim: int, side length of square image
    :param data: entire dataset in memory from which we will pick fragments
    :param label: single integer representing the digit in the cluttered image
    :return: image_out: ndarray size image_dim x image_dim with digit randomly placed with
    8 x 8 patches of images with input noise.
    """

    # Initialize Blank image_out
    image_out = np.zeros([image_dim, image_dim])

    # Randomly Scale image
    h = np.random.randint(low=int(28/2), high=int(28*2))
    w = np.random.randint(low=int(28/2), high=int(28*2))
    digit = zoom(input_image, (h/28, w/28))

    # Randomly choose location in image_out
    x = np.random.randint(low=0, high=image_dim - w)
    y = np.random.randint(low=0, high=image_dim - h)

    # Insert digit into image_out and get max
    image_out[y:y + h, x:x + w] += digit
    max_val = image_out.max()
    
    # Tighten box
    rows = np.sum(image_out, axis=0).round(1)
    cols = np.sum(image_out, axis=1).round(1)
    
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
    if label == 0:  # Faster RCNN regards 0 as background, so change the label for all zeros to 10
        label = 10
    gt_box = [int(left), int(upper), int(right), int(lower), int(label)]
    
    # Track "energy" in gt_box (to prevent clutter insertion)
    energy = np.sum(image_out[upper:lower, left:right])

    # Add in total number of clutter patches
    for j in range(int(image_dim/4)):

        # Randomly select MNIST digit
        index = np.random.choice(len(data))
        fragment = np.reshape(data[index, :], (28, 28))

        # Randomly select patch of selected digit
        px = np.random.randint(low=0, high=28 - 8)
        py = np.random.randint(low=0, high=28 - 8)

        # Randomly choose location to insert clutter
        x = np.random.randint(low=0, high=image_dim - 8)
        y = np.random.randint(low=0, high=image_dim - 8)
            
        # Insert digit fragment  
        image_out[y:(y+8), x:(x+8)] += fragment[py:(py+8), px:(px+8)]
        
        # Don't insert clutter into the true bounding box
        new_energy = np.sum(image_out[upper:lower, left:right])
        if energy != new_energy:
            image_out[y:(y+8), x:(x+8)] -= fragment[py:(py+8), px:(px+8)]
            continue
        
    # Normalize any over-saturated pixels
    image_out = np.clip(image_out, 0, max_val)
        
    # Subtract mean from image and scale to be between -1 and 1
    image_out -= image_out.mean()
    image_out = image_out / np.abs(image_out).max()

    return image_out, gt_box


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
