# Data Files

- `1.csv`-`10.csv`: contains runs from 5 to 4500 neurons and 2 layers obtained on Frodo. In the files we have `execution_time`, `prediction_time` for 10000 images, 
  `By_image` prediction time for each image, `Loss` the value of the loss function after 10 epochs, `Acc` accuracy after 10 epochs.
  **Nvidia driver**: 465.19.01, **CUDA** 11.3, **TF** 2.5.0
- `frodo2.csv`: one run as described above using only one single GPU.
- `frodo3.csv`: one run as described above using three GPUs.
- `f_averages.csv`: this files contain the average of the 10 runs.
- `batch2_<NUMBER>`: with NUMBER =1,2,3. Contains the results of 3 runs on frodo.  
  Running setup : mnist_rpi6.py -n 5,10,50,128,256,512,1024,1500,2048,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000  
  -l 2,4,6,8 
- `batch3_<NUMBER>`: with NUMBER =1,2,3. Fashion mnist results, with the same setup as batch2, to compare the results.