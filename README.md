# tf-Faster-RCNN
A Python 3/TensorFlow implementation of Faster R-CNN ([paper](https://arxiv.org/abs/1506.01497)). See official implementations here:
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
2. Python 3.5+: I recommend Anaconda for your Python distribution and package management. See (3) below.
3. TensorFlow v1.0: See [TensorFlow Installation with Anaconda](https://www.tensorflow.org/install/install_linux#InstallingAnaconda). Install the version that matches your preferred Python version. Instructions for Python 3.6 below:
  ```Shell
  # Create a Conda environment for TensorFlow (defaults to Python 3.6)
  conda create --name tensorflow 
  
  # Activate your environment
  source activate tensorflow
  
  # Install TensorFlow, for Python 3.6 with GPU support
  pip install --ignore-installed --upgrade \
  https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.3.0-cp36-cp36m-linux_x86_64.whl
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
  |--tf-FRC_ROOT
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

#### *Optional: Pre-trained convolutional feature extractor weights*
If you want to use pre-trained weights for the convolutional feature extractor (highly recommended), you have to download those [here](https://github.com/tensorflow/models/tree/master/research/slim#Pretrained). Currently, we have ResNet 50 V1 available; to use it, download the appropriate checkpoint file first.

#### Configure the model
The network architecture and model parameters depend on the kind of data you are trying to process. Most of these are adjustable from the config file. 

Default settings and their descriptions are located at [Lib/faster_rcnn_config.py](https://github.com/kevinjliang/tf-Faster-RCNN/blob/master/Lib/faster_rcnn_config.py). You should not modify this. Instead, write a yaml file, save it under Models/cfgs, and pass it as an argument to your model. See [Models/cfgs/](https://github.com/kevinjliang/tf-Faster-RCNN/tree/master/Models/cfgs) for examples.

In particular, make sure to change the following:
- Point `DATA_DIRECTORY` to your dataset folder (denoted by [YOUR_DATASET] in the earlier file tree). Make this path relative to the [Models/](https://github.com/kevinjliang/tf-Faster-RCNN/tree/master/Models) directory.
- *Optional*: Point `RESTORE_SLIM_FILE` to the location of the checkpoint file you downloaded, if using pre-trained weights for the convolutional feature extractor.
- Change `CLASSES` to a list of the class names in your data. IMPORTANT: Leave the first class as '__background__'
- Update `NUM_CLASSES` to the number of classes in `CLASSES`

The model file you use depends on the data you wish to train on. For something like the simple, single-channeled cluttered MNIST, [Model/faster_rcnn_conv5.py](https://github.com/kevinjliang/tf-Faster-RCNN/blob/master/Models/faster_rcnn_conv5.py) is probably sufficient. More complex, RGB-channeled real data like PASCAL VOC, MS COCO, or ImageNet require a correspondingly more advanced architecture ([example](https://github.com/kevinjliang/tf-Faster-RCNN/blob/master/Models/faster_rcnn_resnet50ish.py)).

Make sure that the number of channels of the input placeholder in the `_data` constructor function matches your data. `faster_rcnn_conv5.py` is defaulted to a single channel (grayscale). `faster_rcnn_resnet50ish.py` is three channels (RGB).

#### Train and Test
To train the model (assuming`faster_rcnn_resnet50ish.py`):
  ```Shell
  python faster_rcnn_resnet50ish.py [arguments]
  ```

Additional arguments (defaults in parentheses; see main() of a model file for additional comments):
- [-n]: Run number (0) - Log files and checkpoints will be saved under this ID number
- [-e]: Epochs (1) - Number of epochs to train the model
- [-r]: Restore (0) - Restore all weights from a specific run number ID (a previous -n) and checkpoint number if equal to 1. Additional specifications in -m and -f (see below)
- [-m]: Model Restore (1) - Specifies which previous run number ID (a previous -n) to restore from
- [-f]: File Epoch (1) - Specifies which epoch checkpoint to restore from
- [-s]: Slim (1) - For models with pre-trained weights, load weights from a downloaded checkpoint if equal to 1
- [-t]: Train (1) - Train the model if equal to 1 (Set this to 0 if you only want to evaluate)
- [-v]: Eval (1) - Evaluate the model if equal to 1
- [-y]: YAML (pascal_voc2007.yml) - Name of the YAML file in the `Models/cfg/` folder to override [faster_rcnn_config.py](https://github.com/kevinjliang/tf-Faster-RCNN/blob/master/Lib/faster_rcnn_config.py) defaults
- [-l]: Learning Rate (1e-3) - Initial Learning Rate (You should actually probably specify this in your YAML, not here)
- [-i]: Visualize (0) - Project output bounding boxes onto your images and save under `Data/[YOUR_DATASET]/Outputs` if equal to 1
- [-g]: GPU (0) - Specify which GPU to use. Input 'all' to use all available GPUs.

Common use cases:
  ```Shell
  # Train model, starting from pre-trained weights
  python faster_rcnn_resnet50ish.py -n 10 -e 20 -y 'myYAML.yml'
  
  # Test model, using a previously trained model (0). Save output images with boxes
  python faster_rcnn_resnet50ish.py -n 11 -r 1 -m 10 -f 18 -t 0 -i 1 -y 'myYAML.yml'
  ```
