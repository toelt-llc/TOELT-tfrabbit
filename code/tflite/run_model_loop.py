#!/usr/bin/env python3
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import sys
import os 

#TODO : main function

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

def run_tflite_model(tflite_file, test_image_indices):
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


tflite_models = []
for dirname, _, filenames in os.walk('./mnist_tflite_models/'):
    for filename in filenames:
        tflite_models.append(os.path.join(dirname, filename))

num_iter = int(sys.argv[1])
inferences = {}

for model in tflite_models:
    print('Model running is : ', model)
    tflite_model = open(model, "rb").read()
    inferences[model]=[]
    i = 0
    for i in range(num_iter):
        #inferences.append(evaluate_model(tflite_model))
        inferences[model].append(evaluate_model(tflite_model))
        i +=1

infdf = pd.DataFrame.from_dict(inferences)
print(infdf)

## Run classic TF part 
classic_inferences = {}

cnn_model = tf.keras.models.load_model('./mnist_models/CNN_classic')
ffnn_model = tf.keras.models.load_model('./mnist_models/FFNN_classic')

start = time.time()
loss, acc = cnn_model.evaluate(test_images, test_labels, verbose=2)
end = time.time()
print('Restored cnn model, accuracy: {:5.2f}%'.format(100 * acc))
print('Inference time : ', round(end-start,2))

start = time.time()
loss, acc = ffnn_model.evaluate(test_images, test_labels, verbose=2)
end = time.time()
print('Restored ffnn model, accuracy: {:5.2f}%'.format(100 * acc))
print('Inference time : ', round(end-start,2))