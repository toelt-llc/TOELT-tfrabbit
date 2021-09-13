import tensorflow as tf 
import numpy as np
import time
import os

from tensorflow import keras

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

inf_time = []
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

def load_data():
    mnist = tf.keras.datasets.mnist
    fash = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images_norm = train_images.astype(np.float32) / 255.0
    test_images_norm = test_images.astype(np.float32) / 255.0

    return train_images_norm, train_labels, test_images_norm, test_labels

model = open("converted_model.tflite", "rb").read()

train_imgs, _, test_imgs, test_labels = load_data()
interpreter = interpret(model, test_imgs)
run_inference(interpreter)

print('Inference time {}'.format(round(inf_time[0],2)))
