#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import time 
import pathlib
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

tflite_models_dir = pathlib.Path("./mnist_tflite_models/")
#tflite_models_dir.mkdir(exist_ok=True, parents=True)

# Data part, may change in the future.
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# CNN
def CNN():
    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28)),
    tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
    ])

    # Train the digit classification model
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True),
                metrics=['accuracy'])
    model.fit(train_images,train_labels,epochs=5, validation_data=(test_images, test_labels))
    
    return model, CNN.__name__

def FFNN():
    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28)),
    tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
    tf.keras.layers.Dense(320),
    tf.keras.layers.Dense(160),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True),
                metrics=['accuracy'])
    model.fit(train_images,train_labels,epochs=1,validation_data=(test_images, test_labels))
    
    # maybe use it for later
    #model.save('saved_model/my_model_FFNN')
    return model, FFNN.__name__

def representative_data_gen():
    """ Necessary for the quant8 part
    """
    for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
        yield [input_value]

def convert(converter, model_name):
    # Converts to a TensorFlow Lite model, but it's still using 32-bit float values for all parameter data.
    tflite_model = converter.convert()
    # Save
    tflite_model_file = tflite_models_dir/(model_name + '_mnist_model.tflite')
    tflite_model_file.write_bytes(tflite_model)
    print('Successfully saved ', tflite_model_file )

    return tflite_model

def convert_quant(converter, model_name):
    # The model is now a bit smaller with quantized weights, but other variable data is still in float format.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model_quant = converter.convert()
    # Save the -default- quantized model:
    tflite_model_quant_file = tflite_models_dir/(model_name + '_mnist_model_quant.tflite')
    tflite_model_quant_file.write_bytes(tflite_model_quant)
    print('Successfully saved ', tflite_model_quant_file)

    return tflite_model_quant

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

def disk_usage():
    print('tflite models sizes : ')
    for _,_,filenames in os.walk(tflite_models_dir):
        #print(filenames)
        for file in filenames:
            print(file, ':', os.stat(os.path.join(tflite_models_dir,file)).st_size)

conv, name_cnn = CNN()
converter_CNN = tf.lite.TFLiteConverter.from_keras_model(conv)

convert(converter_CNN, name_cnn)
convert_quant(converter_CNN, name_cnn)
convert_quant8(converter_CNN, name_cnn)

forw, name_ffnn = FFNN()
converter_FFNN = tf.lite.TFLiteConverter.from_keras_model(conv)

convert(converter_FFNN, name_ffnn)
convert_quant(converter_FFNN, name_ffnn)
convert_quant8(converter_FFNN, name_ffnn)

disk_usage()