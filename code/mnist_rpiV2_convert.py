#!/usr/bin/env python3
import tensorflow as tf
import pandas as pd
import numpy as np
import time 
import pathlib
import sys
import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

tflite_models_dir = pathlib.Path("./progressive_models_lite/")
classic_models_dir = pathlib.Path('./progressive_models')

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Convert whole TF models dir to TFlite part

def representative_data_gen():
    """ Necessary for the quant8 part
    """
    for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
        yield [input_value]

# From train_convert
def convert_quant8(converter, model_name):
    # Convert using integer-only quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model_quant8 = converter.convert()
    # Save the integer quantized model:
    tflite_model_quant8_file = tflite_models_dir/(model_name + '_mnist_model_quant8.tflite')
    tflite_model_quant8_file.write_bytes(tflite_model_quant8)
    print('Successfully saved ', tflite_model_quant8_file )

    return tflite_model_quant8

dic = {} #Save the converting times
for _,_,models in os.walk('./progressive_models/'):
    for model in sorted(models):
        dic[model] = []
        start = time.time()

        to_conv = tf.keras.models.load_model('./progressive_models/'+model)
        converter = tf.lite.TFLiteConverter.from_keras_model(to_conv)
        convert_quant8(converter, model)

        end = time.time()
        dic[model].append(round(end-start,2))

dfres = pd.DataFrame.from_dict(dic)
print(dfres)
dfres.to_csv('./saved_results/fnn_inferences/converting_times.csv')