import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time 

#TODO : main functions, args : loop size, model file

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
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

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    predictions[i] = output.argmax()

    return predictions

def evaluate_model(tflite_file, model_type):
    global test_images
    global test_labels

    test_image_indices = range(test_images.shape[0])
    start = time.time()
    predictions = run_tflite_model(tflite_file, test_image_indices)
    end = time.time()
    accuracy = (np.sum(test_labels== predictions) * 100) / len(test_images)

    print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (
        model_type, accuracy, len(test_images)))
    print('Inference time is : ', end-start)
    return end-start


tflite_model = open('./tflite_models/CNN_8.tflite', "rb").read()

inferences = []
i = 0
for i in range(10):
    inferences.append(evaluate_model(tflite_model, model_type="Quantized"))
    i +=1

print(inferences)


## Test Image

# # Change this to test a different image
# test_image_index = 1
# ## Helper function to test the models on one image
# def test_model(tflite_file, test_image_index, model_type):
#   global test_labels

#   predictions = run_tflite_model(tflite_file, [test_image_index])

#   plt.imshow(test_images[test_image_index])
#   template = model_type + " Model \n True:{true}, Predicted:{predict}"
#   _ = plt.title(template.format(true= str(test_labels[test_image_index]), predict=str(predictions[0])))
#   plt.grid(False)
#   #plt.show()

# test_model(tflite_model, test_image_index, model_type="Quantized")
