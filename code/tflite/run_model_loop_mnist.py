#!/usr/bin/env python3
import tensorflow as tf
import pandas as pd
import numpy as np
import pathlib
import pickle
import time
import sys
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

#TODO : main function

tflite_models_dir = pathlib.Path("./mnist_tflite_models/")
mnist_models_dir = pathlib.Path('./mnist_models')

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

def disk_usage(dir):
    sizes_kb = {}
    print('Models sizes : ')
    for _,_,filenames in os.walk(dir):
        #print(filenames)
        for file in sorted(filenames):
            print(file, ':', os.stat(os.path.join(dir,file)).st_size/1000, 'kb')
            sizes_kb[file] = os.stat(os.path.join(dir,file)).st_size/1000
    return sizes_kb


## Run TFLite 
tflite_models = []
for dirname, _, filenames in os.walk('./mnist_tflite_models/'):
    for filename in sorted(filenames):
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

infdf = pd.DataFrame.from_dict(inferences) # tflite results df
print(infdf)

## Run classic TF part 
cnn_model = tf.keras.models.load_model('./mnist_models/CNN_classic.h5')
ffnn_model = tf.keras.models.load_model('./mnist_models/FFNN_classic.h5')

tf_models = [cnn_model, ffnn_model]
classic_inferences = {} #{'cnn':[],'ffnn':[]}
for model in tf_models:
    classic_inferences[model._name] = []
    i = 0
    for i in range(num_iter):
        start = time.time()
        loss, acc = model.evaluate(test_images, test_labels, verbose=False)
        end = time.time()
        classic_inferences[model._name].append(round(end-start,2))
        i +=1

classic_infdf = pd.DataFrame.from_dict(classic_inferences) # tf results df
print(classic_infdf)

result = pd.concat([infdf, classic_infdf], axis=1)
result.append(result.std(), ignore_index=True)

# Memory usage
litemodels_size = list(disk_usage(tflite_models_dir).values())
models_size = list(disk_usage(mnist_models_dir).values())
sizes_list = litemodels_size + models_size

# The pickle file will contain a list including the combined dataframes + the disk size
name = sys.argv[2]
data = []
data.append(result), data.append(sizes_list)
with open('RPI_inferences_mnist_'+str(num_iter)+name+'.pkl', 'wb') as f:
    pickle.dump(data, f)

result.to_csv('RPI_inferences_mnist_'+str(num_iter)+name+'.csv', index=False)