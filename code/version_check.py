#
# Version Checker
#
# This script check the versions of TensorFlow, cuda, cuDNN and
# the NVIDIA driver.
#
# Version 1.0
# Author: Umberto Michelucci, umberto.michleucci@toelt.ai
# (C) TOELT LLC 2021
#


import subprocess
import sys
import tensorflow as tf

print("----------------------------------------------------")
print("TensorFlow Version:", tf.__version__)
print("----------------------------------------------------")
command = subprocess.run('/usr/local/cuda/bin/nvcc --version', capture_output=True, shell = True    )
sys.stdout.buffer.write(command.stdout)
print("----------------------------------------------------")
command = subprocess.run('nvidia-smi', capture_output=True, shell = True    )
sys.stdout.buffer.write(command.stdout)
print("----------------------------------------------------")
