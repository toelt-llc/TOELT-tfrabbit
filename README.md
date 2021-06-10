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
used is CIFAR-10. The training is performed with `ImageGenerator` with a `batch_size=100`. So for 100 
effective images.

## Sample Results

### Summary

| CPU | GPU | CUDA Version | NVIDIA Driver (min) | VGG19 Time (min) | resnet50 Time (min)|
|-----|-----|------|-----|----|----|
| Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz - 18 Cores |  NVIDIA RTX A6000 GPU 48 Gb Memory | 11.3 | 465.19.01 | 0.38 | 0.41 |
| Intel(R) Xeon(R) CPU @ 2.20GHz |  Tesla T4, 15 Gb Memory | 11.2 | 460.32.03 | 1.05 | 0.70 |
| CPU Intel(R) i9 - 2.3 GHz 8-Core |  None | N/A | N/A | 16.8 | 17.2 |


### Dedicated Deep Learning Linux Server

- NVIDIA RTX A6000 GPU 48 Gb Memory
- Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz - 18 Cores

Software stack
- TensorFlow 2.5
- CUDA 11.3
- NVIDIA Driver 465.19.01

Benchmark Output

    -------------------------------------------------
    Benachmark Results for VGG19

    Elapsed Time (min): 0.3828892230987549
    -------------------------------------------------
    500/500 [==============================] - 25s 43ms/step - loss: 3.0390 - accuracy: 0.1462
    -------------------------------------------------
    Benachmark Results for resnet50

    Elapsed Time (min): 0.4155214587847392
    -------------------------------------------------

### Macbook Pro 16 in (2020)

- CPU Intel(R) i9 - 2.3 GHz 8-Core 

Software Stack
- TensorFlow 2.5


Benchmark Output

        -------------------------------------------------
        Benachmark Results for VGG19

        Elapsed Time (min): 16.789986399809518
        -------------------------------------------------
        500/500 [==============================] - 1029s 2s/step - loss: 2.8604 - accuracy: 0.1865
        -------------------------------------------------
        Benachmark Results for resnet50

        Elapsed Time (min): 17.157415350278217
        -------------------------------------------------
        
### Google Colab

- Intel(R) Xeon(R) CPU @ 2.20GHz
- Tesla T4, 15 Gb Memory

Software stack
- TensorFlow 2.5
- CUDA 11.2
- NVIDIA Driver 460.32.03

        -------------------------------------------------
        Benachmark Results for VGG19

        Elapsed Time (min): 1.0513679345448812
        -------------------------------------------------
        500/500 [==============================] - 42s 72ms/step - loss: 2.9594 - accuracy: 0.1757
        -------------------------------------------------
        Benachmark Results for resnet50

        Elapsed Time (min): 0.7041642387708028
        -------------------------------------------------


