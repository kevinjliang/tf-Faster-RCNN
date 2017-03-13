# tf-Faster-RCNN: *Work in progress (both the code and this README)*
A Python 3.5 + TensorFlow v1.0 implementation of Faster R-CNN ([paper](https://arxiv.org/abs/1506.01497)). See official implementations here:
- [Python + Caffe](https://github.com/rbgirshick/py-faster-rcnn)
- [MATLAB + Caffe](https://github.com/ShaoqingRen/faster_rcnn)

The deep models in this implementation are built on [TensorBase](https://github.com/dancsalo/TensorBase), a minimalistic framework for end-to-end TensorFlow applications created by my good friend and collaborator [Dan Salo](https://github.com/dancsalo). Check it out. [My personal fork](https://github.com/kevinjliang/TensorBase) (whose changes are typically regularly pulled into Dan's) is a submodule of this tf-Faster-RCNN repo.

### Contents
1. [Requirements: Software](#requirements-software)
2. [Installation](#installation)
3. [Repo Organization](#repo-organization) 
4. [Training and Testing](#training-and-testing)


### Requirements: Software
1. Ubuntu 16: I haven't tested it on any other Linux distributions or versions, but there's a chance it might work as is. Let me know if it does!
2. Python 3.5: I recommend Anaconda for your Python distribution and package management. Make sure to have Python 3.5, as 3.6 is not currently supported by TensorFlow. See (3) below.
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


### Installation
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


### Repo Organization
- Data: Scripts for creating, downloading, organizing datasets. For your local copy, the actual data will also reside here
- Development: Experimental code still in development
- Lib: Library functions necessary to run Faster R-CNN
- Models: Runnable files that create a Faster R-CNN class with a specific convolutional network and dataset. Config files for changing model parameters are also here.
- Networks: Neural networks or components that form parts of a Faster R-CNN network


### Training and Testing
If you would like to try training and/or testing the Faster R-CNN network, we currently have models available for translated and cluttered MNIST. Translated MNIST is an MNIST digit embedded into a larger black image. Cluttered MNIST is the same, but with random pieces of other MNIST digits scattered throughout. Both serve as simple datasets for detection, as the algorithm must find the digit and classify it.

To run one of these models (we'll use cluttered MNIST, since it's more interesting):

1. Generate the data:
  ```Shell
  cd $tf-FRC_ROOT/Data/scripts
  
  # Generate images and bounding box data; place it in the folder $tf-FRC_ROOT/Data/data_clutter 
  python MNIST.py
  ```
2. Run the model:
  ```Shell
  cd $tf-FRC_ROOT/Models
  
  # Change flags accordingly
  python faster_rcnn_clutter.py -n [Model num, ex 1] -e [Num of epochs, ex 3]
  ```
  
3. To reload a previously trained model and test
  ```Shell
  python faster_rcnn_clutter.py -r 1 -m [Model num] -f [epoch to restore] -t 0
  ```
