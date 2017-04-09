# tf-Faster-RCNN: *Work in progress (both the code and this README)*
A Python 3.5 + TensorFlow v1.0 implementation of Faster R-CNN ([paper](https://arxiv.org/abs/1506.01497)). See official implementations here:
- [Python + Caffe](https://github.com/rbgirshick/py-faster-rcnn)
- [MATLAB + Caffe](https://github.com/ShaoqingRen/faster_rcnn)

The deep models in this implementation are built on [TensorBase](https://github.com/dancsalo/TensorBase), a minimalistic framework for end-to-end TensorFlow applications created by my good friend and collaborator [Dan Salo](https://github.com/dancsalo). Check it out. [My personal fork](https://github.com/kevinjliang/TensorBase) (whose changes are typically regularly pulled into Dan's) is a submodule of this tf-Faster-RCNN repo.

## Contents
1. [Requirements: Software](#requirements-software)
2. [Installation](#installation)
3. [Repo Organization](#repo-organization) 
4. [Simple Demo](#simple-demo)
5. [Training and Testing Your Data](#training-and-testing-your-data)


## Requirements: Software
1. Ubuntu 16: I haven't tested it on any other Linux distributions or versions, but there's a chance it might work as is. Let me know if it does!
2. Python 3.5: I recommend Anaconda for your Python distribution and package management. See (3) below.
3. TensorFlow v1.0: See [TensorFlow Installation with Anaconda](https://www.tensorflow.org/install/install_linux#InstallingAnaconda). Specifiy Python 3.5 when creating your Conda environment:
  ```Shell
  # Create a Conda environment for TensorFlow v1.0 with Python 3.5
  conda create --name tensorflow python 3.5
  
  # Activate your enviroment
  source activate tensorflow
  
  # Install TensorFlow v1.0, for Python 3.5 with GPU support
  pip install --ignore-installed --upgrade \
  https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.0-cp35-cp35m-linux_x86_64.whl
  ```
4. Some additional python packages you may or may not already have: `cython`, `easydict`, `matplotlib` `scipy`, `Pillow`, `pyyaml`, `tqdm`. These should all be pip installable within your Anaconda environment (pip install [package]):

  ```Shell
  pip install cython easydict matplotlib scipy Pillow pyyaml tqdm 
  ```
5. TensorBase: Tensorbase is used as a submodule, so you can get this recursively while cloning this repo. See [Installation](#installation) below.


## Installation
1. Clone this repository (tf-Faster-RCNN) 
  ```Shell
  # Make sure to clone with --recursive. This'll also clone TensorBase
  git clone --recursive https://github.com/kevinjliang/tf-Faster-RCNN.git
  ```
  
2. We'll call the directory that you cloned tf-Faster-RCNN into `tf-FRC_ROOT`

   *Ignore notes 1 and 2 if you followed step 1 above.*

   **Note 1:** If you didn't clone tf-Faster-RCNN with the `--recursive` flag, then you'll need to manually clone the `TensorBase` submodule:
    ```Shell
    git submodule update --init --recursive
    ```
    **Note 2:** The `TensorBase` submodule needs to be on the `faster-rcnn` branch (or equivalent detached state). This will happen automatically *if you followed step 1 instructions*.

3. Build the Cython modules
  ```Shell
  cd $tf-FRC_ROOT/Lib
  make
  ```


## Repo Organization
- Data: Scripts for creating, downloading, organizing datasets. Output detections are saved here. For your local copy, the actual data will also reside here
- Development: Experimental code still in development
- Lib: Library functions necessary to run Faster R-CNN
- Logs: Holds the tfevents files for TensorBoard, model checkpoints for restoring, and validation/test logs. This directory is created the first time you run a model.
- Models: Runnable files that create a Faster R-CNN class with a specific convolutional network and dataset. Config files for changing model parameters are also here.
- Networks: Neural networks or components that form parts of a Faster R-CNN network


## Simple Demo
If you would like to try training and/or testing the Faster R-CNN network, we currently have a complete model available for cluttered MNIST. Cluttered MNIST is a dataset of images consisting of randomly scaled MNIST digits embedded in a larger image, with random pieces of other MNIST digits scattered throughout. It serves as a simple dataset for detection, as the algorithm must find the digit and classify it. PASCAL VOC and MS COCO on the way.

To run the model on cluttered MNIST:

1. Generate the data:
  ```Shell
  cd $tf-FRC_ROOT/Data/scripts
  
  # Generate images and bounding box data; place it in the folder $tf-FRC_ROOT/Data/clutteredMNIST 
  python MNIST.py
  ```
2. Run the model:
  ```Shell
  cd $tf-FRC_ROOT/Models
  
  # Change flags accordingly (see argparser in main() of Model/faster_rcnn_conv5.py file)
  python faster_rcnn_conv5.py -n [Model num, ex 1] -e [Num of epochs, ex 5]
  ```
  
3. To reload a previously trained model and test
  ```Shell
  # For just mAP and AP performance metrics:
  python faster_rcnn_conv5.py -r 1 -m [Model num] -f [epoch to restore] -t 0

  # To also save test images with overlaid detection boxes as PNGs:
  python faster_rcnn_conv5.py -r 1 -m [Model num] -f [epoch to restore] -t 0 -i 1
  ```

## Training and Testing Your Data
In order to train (and then test) on your own data:

#### Organize your data into the following format:
  ```Shell
  |--tf-Faster-RCNN_ROOT
    |--Data/
      |--[YOUR_DATASET]/
        |--Annotations/
          |--*.txt (Annotation Files: (x1,y1,x2,y2,label))
        |--Images/
          |--*.[png/jpg] (Image files)
        |--Names/
          |--train.txt (List of training data filenames)
          |--valid.txt (List of validation data filenames)
          |--test.txt  (List of testing data filenames)

  ```

Step 1 of the [cluttered MNIST demo](#simple-demo) automatically creates this data and organizes it accordingly, so run the `MNIST.py` script for an example file structure.

#### Configure the model
The network architecture and model parameters depend on the kind of data you are trying to process. Most of these are adjustable from the config file. 

Default settings and their descriptions are located at [Lib/fast_rcnn_config.py](https://github.com/kevinjliang/tf-Faster-RCNN/blob/master/Lib/fast_rcnn_config.py). You should not modify this. Instead, write a yaml file, save it under Models/cfgs, and pass it as an argument to your model. See [Models/cfgs/clutteredMNIST.yml](https://github.com/kevinjliang/tf-Faster-RCNN/blob/master/Models/cfgs/clutteredMNIST.yml) as an example.

In particular, make sure to change the following:
- Point `DATA_DIRECTORY` to your dataset folder (denoted by [YOUR_DATASET] in the earlier file tree). Make this path relative to the [Models/](https://github.com/kevinjliang/tf-Faster-RCNN/tree/master/Models) directory.
- Change `CLASSES` to a list of the class names in your data. IMPORTANT: Leave the first class as '__background__'
- Update `NUM_CLASSES` to the number of classes in `CLASSES`

The model file you use depends on the data you wish to train on. For something like the simple, single-channeled cluttered MNIST, [Model/faster_rcnn_conv5.py](https://github.com/kevinjliang/tf-Faster-RCNN/blob/master/Models/faster_rcnn_conv5.py) is probably sufficient. More complex, RGB-channeled real data like PASCAL VOC, MS COCO, or ImageNet require a correspondingly more advanced architecture ([example](https://github.com/kevinjliang/tf-Faster-RCNN/blob/master/Models/faster_rcnn_resnet50ish.py)).


#### Train and Test
...to be continued (see demo for now)

