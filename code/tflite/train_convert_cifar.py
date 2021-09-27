#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import time 
import pathlib
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

tflite_models_dir = pathlib.Path("./cifar_tflite_models/")
cifar_models_dir = pathlib.Path('./cifar_models')

cifar = tf.keras.datasets.cifar100
(train_images, train_labels), (test_images, test_labels) = cifar.load_data()
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0


# CNN
def CNN():
    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(32,32,3)),
    tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100)
    ])

    model._name = 'cnn'
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True),
                metrics=['accuracy'])
    model.fit(train_images,train_labels,epochs=5, validation_data=(test_images, test_labels))

    print('Saved model : ./cifar_models/CNN_classic.h5')
    model.save('./cifar_models/CNN_classic.h5')

    return model, CNN.__name__

def FFNN():
    #global classes
    #w,l = images.shape[1], images.shape[2]
    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(32, 32, 3)),
    tf.keras.layers.Dense(40),
    #tf.keras.layers.Dense(40),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100)
    ])

    model._name = 'ffnn'
    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True),
                metrics=['accuracy'])
    model.fit(train_images,train_labels,epochs=3,validation_data=(test_images, test_labels))

    print('Saved model :  ./cifar_models/FFNN_classic.h5')
    model.save('./cifar_models/FFNN_classic.h5')

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
    tflite_model_file = tflite_models_dir/(model_name + '_cifar_model.tflite')
    tflite_model_file.write_bytes(tflite_model)
    print('Successfully saved ', tflite_model_file )

    return tflite_model

def convert_quant(converter, model_name):
    # The model is now a bit smaller with quantized weights, but other variable data is still in float format.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model_quant = converter.convert()
    # Save the -default- quantized model:
    tflite_model_quant_file = tflite_models_dir/(model_name + '_cifar_model_quant.tflite')
    tflite_model_quant_file.write_bytes(tflite_model_quant)
    print('Successfully saved ', tflite_model_quant_file)

    return tflite_model_quant

def convert_quant8(converter, model_name):
    # Convert using integer-only quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]#, tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model_quant8 = converter.convert()
    # Save the integer quantized model:
    tflite_model_quant8_file = tflite_models_dir/(model_name + '_cifar_model_quant8.tflite')
    tflite_model_quant8_file.write_bytes(tflite_model_quant8)
    print('Successfully saved ', tflite_model_quant8_file )

    return tflite_model_quant8

def convert_quant16(converter, model_name):
    # The model is now a bit smaller with quantized weights, but other variable data is still in float format.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.target_spec.supported_types = [tf.float16]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model_quant16 = converter.convert()
    # Save the -default- quantized model:
    tflite_model_quant16_file = tflite_models_dir/(model_name + '_cifar_model_quant16.tflite')
    tflite_model_quant16_file.write_bytes(tflite_model_quant16)
    print('Successfully saved ', tflite_model_quant16_file)

    return tflite_model_quant16

def disk_usage(dir):
    print('Models sizes : ')
    for _,_,filenames in os.walk(dir):
        #print(filenames)
        for file in sorted(filenames):
            print(file, ':', os.stat(os.path.join(dir,file)).st_size/1000, 'kb')

conv, name_cnn = CNN()
converter_CNN = tf.lite.TFLiteConverter.from_keras_model(conv)

convert(converter_CNN, name_cnn)
convert_quant(converter_CNN, name_cnn)
convert_quant8(converter_CNN, name_cnn)
convert_quant16(converter_CNN, name_cnn)


forw, name_ffnn = FFNN()
converter_FFNN = tf.lite.TFLiteConverter.from_keras_model(forw)

convert(converter_FFNN, name_ffnn)
convert_quant(converter_FFNN, name_ffnn)
convert_quant8(converter_FFNN, name_ffnn)
convert_quant16(converter_FFNN, name_ffnn)

disk_usage(tflite_models_dir)
disk_usage(cifar_models_dir)