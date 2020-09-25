# Introduction

This repository holds the code for running MM-HAND model as proposed under
<b>*MM-HAND: 3D-Aware Multi-Modal Guided Hand Generation for Pose Data Augmentation*<b>
submitted to ACM-MM 2020 conference.



# Quick Start
## Environment
We tested our code on Ubuntu 19.10, with CUDA 10.1 and Pytorch v1.4.0
1. clone the current repo
```
$ git clone https://github.com/ScottHoang/mm-hand.git
$ cd ./mm-hand
```
2. create a new pytorch enviroment
3. Install [Pytorch](https://pytorch.org/).
4. Install NVIDIA's APEX following the [official link](https://github.com/NVIDIA/apex). Don't clone NVIDIA repo in our current directory.
5. Install dependencies
```
$ pip install -r requirements.txt
```

## Data
1. create a datasets folder. Assuming the current directory is this repo
```
$ mkdir ./datasets
```
2. Download [Rendered-Hand-Pose dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/RenderedHandposeDataset.en.html)
3. Download [Stereo-Tracking dataset](https://github.com/zhjwustc/icip17_stereo_hand_pose_datasetorchvision==0.5.0)
4. unzip the datasets
5. run
```
$ python ./tools/create_STB_DB.py [Path to downloaded STB dataset] ./datasets/stb_dataset 256
$ python ./tools/create_RHD_DB.py [PATH to downloaded RHD dataset] ./datasets/rhd_dataset 256
```
## Run Script
<b>Be sure to read the options avaiable within scripts<b>
```
$ bash ./scripts/mm-train-ratio.sh
```
# Options
## Definition
## script setup
