# DUGMA: Dynamic Uncertainty-Based Gaussian Mixture Alignment

This repository contains the code for "[DUGMA: Dynamic Uncertainty-Based Gaussian Mixture Alignment](https://arxiv.org/abs/1803.07426)" paper (3DV 2018) by [Can Pu](https://github.com/Canpu999), Nanbo Li, Radim Tylecek, Robert B Fisher. For the speed-up version, thank Nicola Strisciuglio for his [implementation](https://gitlab.com/nicstrisc/fast-dugma). 

### Citation
```
@inproceedings{pu2018dugma,
  title={DUGMA: Dynamic Uncertainty-Based Gaussian Mixture Alignment},
  author={Pu, Can and Li, Nanbo and Tylecek, Radim and Fisher, Bob},
  booktitle={2018 International Conference on 3D Vision (3DV)},
  pages={766--774},
  year={2018},
  organization={IEEE}
}

```

## Contents

1. [Introduction](#introduction)
2. [Usage](#usage)
3. [Contacts](#contacts)

## Introduction

Accurately registering point clouds from a cheap low resolution sensor is a challenging task. Existing rigid registration methods failed to use the physical 3D uncertainty distribution of each point from a real sensor in the dynamic alignment process. It is mainly because the uncertainty model for a point is static and invariant and it is hard to describe the change of these physical uncertainty models in different views. Additionally, the existing Gaussian mixture alignment architecture cannot efficiently implement these dynamic changes.

This paper proposes a simple architecture combining error estimation from sample covariances and dynamic global probability alignment using the convolution of uncertainty-based Gaussian Mixture Models (GMM). Firstly, we propose an efficient way to describe the change of each 3D uncertainty model, which represents the structure of the point cloud better. Unlike the invariant GMM (representing a fixed point cloud) in traditional Gaussian mixture alignment, we use two uncertainty-based GMMs that change and interact with each other in each iteration. In order to have a wider basin of convergence than other local algorithms, we design a more robust energy function by convolving efficiently the two GMMs over the whole 3D space. Tens of thousands of trials have been conducted on hundreds of models from multiple datasets to demonstrate the proposed method’s superior performance compared with the current state-of-the-art methods. 


## Usage

### Dependencies
Please install cuda driver and cuda toolkit and matlab. The environment for our computer:
- Ubuntu 16.04
- gcc version 4.9.3
- g++ version 4.9.3
- Matlab2016b(or higher version)
- Hardware: GTX 1080 Ti
- GPU Driver Version: 384.90
- cuda toolkit 8.0


### How to use it
1 Open matlab and run file 'compile.m'


2 run file 'main_simulation.m'   or   run file 'main_kinect_real_application.m'


### Use your own data
If you want to use your own data, please use the function "DUGMA.m". The current version of DUGMA is a little slow. Thus, we suggest to use the point clouds with 6K points or less to test this algorithm. 





## Contacts
can.pu@amigaga.com

Any discussions or concerns are welcomed!
