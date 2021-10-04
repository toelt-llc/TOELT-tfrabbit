#!/usr/bin/env python3


### In this version, the goal is to try different sizes of models (FNN at first) until it crashes on RPI. 
#   And when it does, does running it in its tflite version also crash ? 'Can we run larger networks thanks to tflite' 

# Requirements : a training script, to have the pretained model and it's tflite version. 
#
# Dataset : mnist

import tensorflow as tf
import pandas as pd
import numpy as np
import pathlib
import pickle
import time
import sys
import os 

tflite_models_dir = pathlib.Path("./progressive_models_lite/")
classic_models_dir = pathlib.Path('./progressive_models')

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

def run_tflite_model(tflite_file, test_image_indices):
    # From run_model_loop
    global test_images

    # Initialize 
    interpreter = tf.lite.Interpreter(model_content=(tflite_file))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = np.zeros((len(test_image_indices),), dtype=int)
    for i, test_image_index in enumerate(test_image_indices):
        test_image = test_images[test_image_index]

        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point

        #test_image = test_image.astype(input_details['dtype'])
        test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()

        #print(interpreter.get_tensor(output_details["index"]))
        output = interpreter.get_tensor(output_details["index"])[0]
        predictions[i] = output.argmax()

    return predictions

def evaluate_model(tflite_file):
    # From run_model_loop
    global test_images
    global test_labels

    test_image_indices = range(test_images.shape[0])
    start = time.time()
    predictions = run_tflite_model(tflite_file, test_image_indices)
    end = time.time()
    accuracy = (np.sum(test_labels== predictions) * 100) / len(test_images)

    #print('Model accuracy is %.4f%% (Number of test samples=%d)' % (accuracy, len(test_images)))
    print('Inference time is : ', round(end-start,2))
    return round(end-start,2)

## Run classic TF part 
fnn_model = tf.keras.models.load_model('./progressive_models/FFNN_classic.h5')

# From run model loop
start = time.time()
loss, acc = fnn_model.evaluate(test_images, test_labels, verbose=False)
end = time.time()
eval_time = round(end-start,2)

# From mnist_rpi9

# Timed inference
size = 1000
test_sample = test_images[:size]
start_test = time.time()
fnn_model.predict(test_sample)
end_test = time.time()
inf_time = round(end_test-start_test, 2)        # Total inference time
inf_img = round((end_test-start_test)/size, 4 ) # Inference time for each image
