import tensorflow as tf 
import numpy as np
import time
import os

from tensorflow import keras

# Imports a saved .tflite model, and runs it on the given data
# Requires a dataset 
# ! Requires to be trained on the same data used here

inf_time = []
def interpret(model, test_set):
    start_int = time.time()
    interpreter = tf.lite.Interpreter(model_content=model)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    #print(input_details)
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

    inf_time.append(end_int-start_int)
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

def load_data():
    mnist = tf.keras.datasets.mnist
    fash = tf.keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images_norm = train_images.astype(np.float32) / 255.0
    test_images_norm = test_images.astype(np.float32) / 255.0
    # Keep in my mid the changes over the training set for the original model, ie you may loose accuracy
    #test_images_norm = test_images_norm.astype(np.uint8)

    return train_images_norm, train_labels, test_images_norm, test_labels

# Data
train_imgs, _, test_imgs, test_labels = load_data()

# TF Inference
classic_model = tf.keras.models.load_model('./saved_model/my_model_CNN')
_, eval = classic_model.evaluate(test_imgs, test_labels)
start = time.time()
preds = classic_model.predict(test_imgs)
dec = np.argmax(preds, axis=1)
end = time.time()
inftime = end-start
print("Classic TF test acc:", round(eval, 3))
print("Classic TF inference time:", round(inftime, 3))
# 2
start = time.time()
preds = classic_model.predict(test_imgs)
dec = np.argmax(preds, axis=1)
end = time.time()
inftime = end-start
print("Classic TF test acc:", round(eval, 3))
print("Classic TF inference time:", round(inftime, 3))

# TF Lite Inference
lite_model8 = open("./tflite_models/test_8.tflite", "rb").read()
lite_model16 = open("./tflite_models/test_16.tflite", "rb").read()
lite_model_raw = open("./tflite_models/test_raw.tflite", "rb").read()

interpreter8 = interpret(lite_model8, test_imgs)
run_inference(interpreter8)
# run_inference(interpreter8)
# run_inference(interpreter8)
interpreter16 = interpret(lite_model16, test_imgs)
run_inference(interpreter16)
interpreter16 = interpret(lite_model16, test_imgs)
run_inference(interpreter16)
interpreter16 = interpret(lite_model16, test_imgs)
run_inference(interpreter16)

interpreter_raw = interpret(lite_model_raw, test_imgs)
run_inference(interpreter_raw)
print('Interpret time 8 {}'.format(round(inf_time[0],2)))
print('Inference time 8 {}'.format(round(inf_time[1],2)))
#print('Interpret time 16 {}'.format(round(inf_time[2],2)))
print('Inference time 16 {}'.format(round(inf_time[3],2)))
print('Inference time 162 {}'.format(round(inf_time[4],2)))
print('Inference time 163 {}'.format(round(inf_time[5],2)))
#print('Interpret time raw {}'.format(round(inf_time[4],2)))
print('Inference time raw {}'.format(round(inf_time[7],2)))
#print(test_imgs.dtype)
