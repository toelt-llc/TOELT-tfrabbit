import subprocess
import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
command = subprocess.run('/usr/local/cuda/bin/nvcc --version', capture_output=True, shell = True)
sys.stdout.buffer.write(command.stdout)
