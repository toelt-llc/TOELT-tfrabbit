## First benchmark 
The first benchmark `mnist_rpi.py` can be used to verifiy the installation of TensorFlow, and if a simple network can be launched on it.

The dataset used is the popular mnist. The file mnist_rpi.py contains a sequential model made of 2 simple Dense layers. 
The training data consist of 60000 images, the test of 10000. 

### Installation 
Tensorflow 2.0 is not available for RaspberryPis using the normal pip installer but solutions exist. 
The solution used here is a cross-compiling Python wheel (a type of binary Python package) for Raspberry Pi. 
Installing Tensorflow requires some extra steps on the Pi's ARM architecture, I used this nice [tutorial video](https://www.youtube.com/watch?v=GNRg2P8Vqqs) to do so.

**Install** in venv : (based on [this guide](https://github.com/PINTO0309/Tensorflow-bin/#usage) )  
`$ sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev`  
`$ python3 -m pip install keras_applications==1.0.8 --no-deps`  
`$ python3 -m pip install keras_preprocessing==1.1.0 --no-deps`  
`$ python3 -m pip install h5py==2.9.0`  
`$ sudo apt-get install -y openmpi-bin libopenmpi-dev`  
`$ sudo apt-get install -y libatlas-base-dev`  
`$ python3 -m pip install -U six wheel mock`  

Picked TF 2.4.0 from [tensorflow-on-arm](https://github.com/lhelontra/tensorflow-on-arm/releases)  
`$ wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v2.4.0/tensorflow-2.4.0-cp37-none-linux_armv7l.whl`  
`$ python3 -m pip uninstall tensorflow`  
`$ python3 -m pip install tensorflow-2.0.0-cp37-none-linux_armv7l.whl`  

To run on RaspberryPi, it was also necessary to change Numpy version to 1.20.0
`$ python3 -m pip uninstall numpy`
`$ python3 -m pip uninstall numpy==1.20.0`