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
