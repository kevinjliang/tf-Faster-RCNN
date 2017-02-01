# tf-Faster-RCNN: *Work in progress (both the code and this README)*
A Python + TensorFlow implementation of Faster R-CNN ([paper](https://arxiv.org/abs/1506.01497)). See offical implementations here:
- [Python + Caffe](https://github.com/rbgirshick/py-faster-rcnn)
- [Matlab + Caffe](https://github.com/ShaoqingRen/faster_rcnn)

The deep models in this implementation are built on [TensorBase](https://github.com/dancsalo/TensorBase), a minimalistic framework for end-to-end TensorFlow applications created by my good friend and collaborator Dan Salo. Check it out. [My personal fork](https://github.com/kevinjliang/TensorBase) (whose changes are typically regularly pulled into Dan's) is a submodule of this tf-Faster-RCNN repo.

### Contents
1. [Requirements: Software](#requirements-software)
3. [Installation](#installation)


### Requirements: Software
1. TensorFlow (obviously): I recommend Anaconda for your Python distribution and package management. See [TensorFlow Installation with Anaconda](https://www.tensorflow.org/get_started/os_setup#anaconda_installation) 
2. Some additional python packages you may or may not already have: `cython`, `easydict`, `scipy`, `pickle`, `Pillow`, `tqdm`. These should all be pip installable within your Anaconda environment (pip install [package]) 
3. TensorBase: Tensorbase is used as a submodule, so you can get this recursively while cloning this repo. See [Installation](#installation) below.


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
  
4. Set up Data (more detailed instructions coming soon...)
