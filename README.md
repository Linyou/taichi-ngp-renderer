# Taichi NGP Renderer

[![License](https://img.shields.io/badge/license-Apache-green.svg)](LICENSE)

This is an [Instant-NGP](https://github.com/NVlabs/instant-ngp) renderer implemented using [Taichi](https://github.com/taichi-dev/taichi), written entirely in Python. **No CUDA!** This repository only implemented the rendering part of the NGP but is more simple and has a lesser amount of code compared to the original (Instant-NGP and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)).

<!-- It is fun to write this code that implements even just the rendering part of the NGP, especially when you compare it original repo, which is a hundred lines verse thousands (Instant-NGP and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)). -->

<p align="center">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/example.png", width="24%">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/lego_depth.png", width="24%">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/interaction.gif", width="24%">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/samples.gif", width="24%">
  <br>
</p>

## Installation

Install Taichi to get started:

```bash
pip install -i https://pypi.taichi.graphics/simple/ taichi-nightly
```
> **Note**
> There is a bug in Taichi codegen that a `uint32` value can not properly work on a modulo operation. You have to install nightly version in which they have this bug fixed. See detail in issue [#6118](https://github.com/taichi-dev/taichi/issues/6118)

Clone this repository and install the required package:

```bash
git clone https://github.com/Linyou/taichi-ngp-renderer.git
python -m pip install -r requirement.txt
```

## Description

This repository only implemented the forward part of the **Instant-NGP**, which include:

- Rays intersection with bounding box: `ray_intersect()`
- Ray marching strategic: `raymarching_test_kernel()`
- Spherical harmonics encoding for ray direction: `dir_encode()`
- Hash table encoding for 3d coordinate: `hash_encode()`
- **Fully Fused MLP** using shared memory: `sigma_layer()`, `rgb_layer()`
- Volume rendering: `composite_test()`

<!-- Since this repository only implemented the rendering code, so you have to load trained parameters from a pre-trained scene (This repository provides a lego scene). -->

However, there are some differences compared to the original:

###### Missing function

- Taichi is currently missing the `frexp()` method, so I have to use a hard-coded scale of 0.5. I will update the code once **Taichi** supports this function.

###### Fully Fused MLP

- Instead of having a single kernel like **tiny-cuda-nn**, this repo use separated kernel `sigma_layer()` and `rgb_layer()` because the shared memory size that **Taichi** currently allow is `48KB` as issue [#6385](https://github.com/taichi-dev/taichi/issues/6385) points out, it could be improved in the future.
- In the **tiny-cuda-nn**, they use TensorCore for `float16` multiplication, which is not an accessible feature for Taichi, so I directly convert all the data to `ti.float16` to speed up the computation.

## GUI

This code supports real-time rendering GUI interactions with less than 1GB VRAM:

- Control the camera with the mouse and keyboard
- Changing the number of samples for each ray while rendering
- Rendering with different resolution
- Support video recording and export video and snapshot (Please install [ffmpeg](https://docs.taichi-lang.org/docs/export_results#install-ffmpeg-on-windows))
- Up to 23 fps on a 3090 GPU at 800 $\times$ 800 resolution

> In order to get 23 fps speed without damaging the visual quality, the  `T_threshold` and `max_samples` is set to 0.1 and 50, respectively.

<!-- `T_threshold`: Stop rendering when transparent vaule is below this threshold.
`max_samples`: Maximum samples for each ray. -->

Run `python taichi_ngp.py --gui --scene lego` to start the GUI. This repository provided eight pre-trained NeRF synthesis scenes: _Lego, Ship, Mic, Materials, Hotdog, Ficus, Drums, Chair_

<p align="center">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/lego.png", width="24%">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/ship.png", width="24%">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/mic.png", width="24%">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/materials.png", width="24%">
  <br>
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/hotdog.png", width="24%">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/ficus.png", width="24%">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/drums.png", width="24%">
  <img src="https://raw.githubusercontent.com/Linyou/taichi-ngp-renderer/main/assets/chair.png", width="24%">
  <br>
</p>

Running `python taichi_ngp.py --gui --scene <name>` will automatically download pre-trained model `<name>` in the `./npy_file` folder. Please check out the argument parameters in `taichi_ngp.py` for more options.

## Custom scene

You can train a new scene with [ngp_pl](https://github.com/kwea123/ngp_pl) and save the pytorch model to numpy using `np.save()`. After that, use the `--model_path` argument to specify the model file.

## Todo

- [ ] Refactor to separate modules
- [ ] Support Vulkan backend
- [ ] Support real scenes

...
