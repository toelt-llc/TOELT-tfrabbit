# tfrabbit

Version 1.0

Author: Umberto Michelucci

This repository contains the benchmarking code for TensorFlow

## Benchmarking 

A simple benchmarking script is `resnet_benchmark1.py`. To use it you need to:

- `git clone `
- `cd tfrabbit`
- `cd code`
- `python resnet_benchmark1.py`

This will gives the time needed to train 1 epoch with two networks: VGG19 and resnet50.
Please note that training on a CPU may take 20-30 minutes for each network so be warned. The dataset
used is CIFAR-10. The training is performed with `ImageGenerator`.

## Sample Results

### Dedicated Deep Learning Linux Server

- NVIDIA RTX A6000 GPU 48 Gb Memory
- Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz - 18 Cores

Software stack
- TensorFlow 2.5
- CUDA 11.3
- NVIDIA Driver 465.19.01

Benchmarking Output

    -------------------------------------------------
    Benachmark Results for VGG19

    Elapsed Time (min): 0.3828892230987549
    -------------------------------------------------
    500/500 [==============================] - 25s 43ms/step - loss: 3.0390 - accuracy: 0.1462
    -------------------------------------------------
    Benachmark Results for resnet50

    Elapsed Time (min): 0.4155214587847392
    -------------------------------------------------

