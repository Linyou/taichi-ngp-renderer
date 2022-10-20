# Taichi NGP Renderer

This is an [Instant-NGP](https://github.com/NVlabs/instant-ngp) renderer implemented using [Taichi](https://github.com/taichi-dev/taichi). It is fun to write this very simple code that allows me to implement even just the rendering part of the NGP, especially when you compare it original repo, which is a hundred lines verse thousands (Instant-NGP and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)).

## Description

This repository only implemented the forward part of the **Instant-NGP**, which include:

- Rays intersection with bounding box: `ray_intersect()`
- Ray marching strategic: `raymarching_test_kernel()`
- Spherical harmonics encoding for ray direction: `dir_encode()`
- Hash table encoding for 3d position: `hash_encode()`
- Fully fuse mlp using shared memory: `sigma_layer()`, `rgb_layer()`
- Volume rendering: `composite_test()`

Since this repository only implemented the rendering code, so you have to load trained parameters from a pre-trained scene (This repository provides a lego scene).

However, there are some differences compared to the original:

- Taichi currently missing `frexp()` method, so I have to use a hard-coded scale which is 0.5
- The fully fuse mlp is not exactly the same as **tiny-cuda-nn**. Instead of having a single kernel, this repo use separated kernel `sigma_layer()` and `rgb_layer()` because the shared memory that **Taichi** currency allow is 48KB as issue [#6385](https://github.com/taichi-dev/taichi/issues/6385) points out.

## GUI

This code supports real-time rendering with GUI interactions:

- Control the camera with the mouse and keyboard
- Rendering with different numbers of samples for each ray
- Up to 17 fps on a 3090 GPU at 800 $\times$ 800 resolution (using default pose)

Simply run `python taichi_ngp.py --gui` to start the GUI. For more options, please check out the argument parameters in `taichi_ngp.py`.

## Custom scene

You can train a new scene with [ngp_pl](https://github.com/kwea123/ngp_pl), and save the pytorch model to numpy using `np.save()`.

## Todo

- [ ] Refactor to separate modules
- [ ] Support resizing the window, and changing the resolution in rendering
- [ ] Support Vulkan backend
