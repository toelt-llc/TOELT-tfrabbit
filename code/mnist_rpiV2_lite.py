#!/usr/bin/env python3
import tensorflow as tf
import pandas as pd
import numpy as np
import time
import pathlib
import sys
import os

tflite_models_dir = pathlib.Path("./progressive_models_lite/")
classic_models_dir = pathlib.Path('./progressive_models')

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# From model loop
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
        start = time.time()
        interpreter.invoke()
        end=time.time()

        #print(interpreter.get_tensor(output_details["index"]))
        output = interpreter.get_tensor(output_details["index"])[0]
        predictions[i] = output.argmax()

    return predictions, round(end-start,2)

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

# Run TFLite part

# From model loop
## Run TFLite
tflite_models = []
#for dirname, _, filenames in os.walk('./progressive_models_lite/'):
for dirname, _, filenames in os.walk('./progressive_models_lite_kept/'):
    for filename in sorted(filenames):
        tflite_models.append(os.path.join(dirname, filename))

num_iter = 1
#num_iter = int(sys.argv[1])
inferences = {}

for model in tflite_models:
    print('Model running is : ', model)
    tflite_model = open(model, "rb").read()
    inferences[model[26:40]]=[]
    i = 0
    for i in range(num_iter):
        inferences[model[26:40]].append(round(evaluate_model(tflite_model),2))
        inferences[model[26:40]].append(round(evaluate_model(tflite_model)/test_images.shape[0],4))
        i +=1

print('To run with arg (device)')
device = sys.argv[1]
# Results
print(inferences)
idx = ['Total Inf', 'Inf / Img']
infdf = pd.DataFrame.from_dict(inferences)
infdf.index = idx
infdf.to_csv('./saved_results/fnn_inferences/mnist_rpiv2lite'+device+'.csv')
