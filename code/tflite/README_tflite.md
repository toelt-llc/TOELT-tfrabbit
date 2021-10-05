## This is the TFLite test directory
Dedicated to quantization experiments, using tf and tflite models and libraries  

### Notebooks 
`example` runs a TFLite example from the site, not really optimized for what we want to do here
`model_maker` is an example run of how to build and create a TFLite inference with the 'maker' library
`tflite_exploration` also uses model_maker, with an image classifier example and some tests

`tflite_converter` is the main notebook, creates a TF model (saves it and reloads it) to then convert it to TFLite
The TFLite invoker is then used with some mnist_data. So far the notebook succesfully translates and uses a TFLite interpreter.


### Scripts
`train_convert`: to be used before run_model_loop, trains tflite models to be then executed on the RPI. Each dataset test has its own training script.  
`run_model_loop`: currently runs a tflite model, over a chosen loop size. runs the inference without changing the inout size, therefore gaining time. Each dataset tested has it own running loop script.  
`stl10_load.py`: used to download and process the stl10 archive.  

`resnet.py`: exemple of resnet implementation and single prediction. (To move)  

### Results, csv and pkl  
May have to be moved to results dir later. Or create a new one in this tflite part.  
The csv files contain the inference time over the same amount (10000) of test examples for each dataset.  
The number in the filename is the loop size, to give a confidence interval.  
The .pkl contains a list : the result dataframe(like in the csv), + the measured size of all the models on disk (for comparison).  




#### Outdated
`train_model_CNN.py`: trains a convolutional NN, measures its inference time, then convert and run it as TFLite. Does flaot32, float16 and int8 quant. Saves TF model under './saved_model/my_model_CNN'. Saves TFLite models under './tflite_models'  
`train_model_FFNN.py` : similar, but with a different model.   
Both these models use the mnist dataset. 
`read_model.py`: To run after a train_model script. Imports a saved .tflite model, and runs it on the given data, requires to load a dataset due to how TFLite works. Requires to be trained on the same data used in train_model.
`tfvs_tflite.py`