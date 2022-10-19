# taichi-ngp-renderer

This is an **Instant-NGP** renderer implemented using **taichi**. It is fun to write this very simple code that allows me to implement even just the rendering part of the **NGP**, especially when you compare it original repo, which is a hundred lines verse thousands.

## Description

This repository only implemented the forward part of the **Instant-NGP**, which include:

1. Rays intersection with bounding box: `ray_intersect()`
2. Ray marching strategic: `raymarching_test_kernel()`
3. Spherical harmonics encoding for ray direction: `dir_encode()`
4. Hash table encoding for 3d position: `hash_encode()`
5. Fully fuse mlp using shared memory: `sigma_layer()`, `rgb_layer()`
6. Volume rendering: `composite_test()`

Since this repository only implemented the rendering code, so you have to load trained parameters from a pre-trained scene (This repository provides a lego scene). 

## Feature

Most importantly, This code supports real-time rendering with GUI interactions:

1. Adjust the camera by mouse control
2. Rendering with different numbers of samples for each ray
3. Up to 17 fps on a 3090 GPU at $800 \times 800$ resolution

## Usage

Simple run `python taichi_ngp.py --gui` to start the GUI. For more options, please check out the argument parameter in `taichi_ngp.py`.