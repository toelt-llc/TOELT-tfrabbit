#!/usr/bin/env python3
import tensorflow as tf
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

# from train_convert
def FFNN():
    #global classes
    #w,l = images.shape[1], images.shape[2]
    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28)),
    tf.keras.layers.Reshape(target_shape=(28,28, 1)),
    tf.keras.layers.Dense(40),
    #tf.keras.layers.Dense(40),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
    ])
    model._name = 'ffnn'

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True),
                metrics=['accuracy'])
    model.fit(train_images,train_labels,epochs=3,validation_data=(test_images, test_labels))
    print('Saved model :  ./mnist_models/FFNN_classic.h5')
    model.save('./mnist_models/FFNN_classic.h5')

    return model, FFNN.__name__