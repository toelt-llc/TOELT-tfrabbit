#!/usr/bin/env python3
import tensorflow as tf
import pandas as pd
import numpy as np
import time 
import pathlib
import pickle
import sys
import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

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
    #print('Inference time is : ', round(end-start,2))
    return accuracy

# Run all kept h5 models, to have an idea on accuracy changes.

dic = {} #Save the converting times
for _,_,models in os.walk('./progressive_models_kept/'):
    for model in sorted(models):
        dic[model] = []
        start = time.time()
        loaded_model = tf.keras.models.load_model('./progressive_models_kept/' + model)
        loss,acc = loaded_model.evaluate(test_images, test_labels)
        # loss = round(eval[0],2)
        # acc = round(eval[1],2)
        end = time.time()

        dic[model].append(acc)
        #dic[model].append(round(end-start,2))

# From model loop
## Run TFLite
tflite_models = []
for dirname, _, filenames in os.walk('./progressive_models_lite_kept/'):
    for filename in sorted(filenames):
        tflite_models.append(os.path.join(dirname, filename))

num_iter = 1
inferences = {}

for model in tflite_models:
    print('Model running is : ', model)
    tflite_model = open(model, "rb").read()
    inferences[model[26:40]]=[]
    i = 0
    for i in range(num_iter):
        inferences[model[26:40]].append(evaluate_model(tflite_model))
        #inferences[model[26:40]].append(round(evaluate_model(tflite_model)/test_images.shape[0],4))
        i +=1

dfres = pd.DataFrame.from_dict(dic)
print(dfres)
dfres.to_csv('./saved_results/fnn_inferences/kept_accuracy_h5.csv')
with open('./saved_results/fnn_inferences/kept_accuracy_dict.pkl', 'wb') as f:
    pickle.dump(dic, f)


dflite = pd.DataFrame.from_dict(inferences)
dflite.to_csv('./saved_results/fnn_inferences/kept_accuracy_lite.csv')
with open('./saved_results/fnn_inferences/lite_accuracy_dict.pkl', 'wb') as f:
    pickle.dump(dflite, f)