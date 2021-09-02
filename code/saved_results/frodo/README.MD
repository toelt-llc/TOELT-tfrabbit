# Data Files

- `1.csv`-`10.csv`: contains runs from 5 to 4500 neurons and 2 layers obtained on Frodo. In the files we have `execution_time`, `prediction_time` for 10000 images, 
  `By_image` prediction time for each image, `Loss` the value of the loss function after 10 epochs, `Acc` accuracy after 10 epochs.
  **Nvidia driver**: 465.19.01, **CUDA** 11.3, **TF** 2.5.0
- `frodo2.csv`: one run as described above using only one single GPU.
- `frodo3.csv`: one run as described above using three GPUs.
- `f_averages.csv`: this files contain the average of the 10 runs.
- `batch2_<NUMBER>`: with NUMBER =1,...10. It will contain the results of 10 runs on frodo with a number of neurons up to 10000, and a number of layers up to 8. 
  There will be 10 of those files when the runs are finalized.
