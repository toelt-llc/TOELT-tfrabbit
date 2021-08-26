## This is the TFLite test directory

`example` runs a TFLite example from the site, not really optimized for what we want to do here
`model_maker` is an example run of how to build and create a TFLite inference with the 'maker' library
`tflite_exploration` also uses model_maker, with an image classifier example and some tests

`tflite_converter` is the main notebook, creates a TF model (saves it and reloads it) to then convert it to TFLite
The TFLite invoker is then used with some mnist_data. So far the notebook succesfully translates and uses a TFLite interpreter.