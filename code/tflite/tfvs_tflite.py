#!/usr/bin/env python3

# if -r : retrain

# 1. Load model(s) and the data
# 2. Inference tf version
# 3. Inference tflite version 
# -> requires a load 
# 4. Profit 


import tensorflow as tf 
import numpy as np
import time

from tensorflow import keras

saved_model1 = './model1_mnist/'
saved_model2 = './model2_mnist/'

# Currently only working when the model is saved & loaded using the keras api, rather than the tf.saved_model api
saved_model2_keras = './model2_keras/'
reloaded_model = tf.keras.models.load_model(saved_model2_keras)


def load_data():
    mnist = tf.keras.datasets.mnist
    fash = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images_norm = train_images.astype(np.float32) / 255.0
    test_images_norm = test_images.astype(np.float32) / 255.0

    return train_images_norm, test_images_norm, test_labels


# tflite conv + inference 
inf_time = []

def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(train_imgs).batch(1).take(1000):
        yield [input_value]

def conv_int8(model_path):
    start_conv = time.time()
    # TODO set option to read it from saved model or from existing model
    # for now : uses keras (existing model)
    # converter1 = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter1 = tf.lite.TFLiteConverter.from_keras_model(model_path)
    converter1.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter1.optimizations = [tf.lite.Optimize.DEFAULT]
    converter1.representative_dataset = representative_data_gen
    converter1.inference_input_type = tf.uint8
    converter1.inference_output_type = tf.uint8

    quant_model = converter1.convert()
    end_conv = time.time()
    inf_time.append(end_conv-start_conv)

    return(quant_model)


def interpret(model, test_set):
    #start_int = time.time()
    interpreter = tf.lite.Interpreter(model_content=model)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_type = interpreter.get_input_details()[0]['dtype']
    # Quantization parameters : 
    input_scale, input_zero_point = input_details[0]["quantization"]

    # whole set, currently only supports test_set as an array
    interpreter.resize_tensor_input(input_index=input_details[0]['index'], tensor_size=np.shape(test_set))
    interpreter.allocate_tensors()

    # 8bit quantization approximation
    test_images_q = test_set / input_scale + input_zero_point
    test_images_q = np.reshape(test_images_q.astype(input_type), np.shape(test_set)) # wordy line

    # Loading into the tensor
    interpreter.set_tensor(input_details[0]['index'], test_images_q)
    end_int = time.time()

    #inf_time.append(end_int-start_int)
    return interpreter

def run_inference(interpreter):
    output_details = interpreter.get_output_details()
    start_inf = time.time()
    interpreter.invoke()
    end_inf = time.time()
    inference = interpreter.get_tensor(output_details[0]['index'])
    predictions = np.argmax(inference, axis=1)

    inf_time.append(end_inf-start_inf)
    print('Quantized model accuracy : ', (predictions == test_labels).mean())

    return predictions


train_imgs, test_imgs, test_labels = load_data()

# Normal TF
pred_time = 0
start_pred = time.time()
pred = reloaded_model.predict(test_imgs)
res_pred = np.argmax(pred, axis=1)
end_pred = time.time()

pred_time = round(end_pred-start_pred, 2)


# TFLite 

quanted_model = conv_int8(reloaded_model)
interpreter = interpret(quanted_model, test_imgs)
run_inference(interpreter)

# Output 
print('Conversion time : {}, Inference time {}'.format(round(inf_time[0],2), round(inf_time[1],2)))
print('Classic model accuracy : ', (res_pred == test_labels).mean() )
print('With inference time : ', pred_time)