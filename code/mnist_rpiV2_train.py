#!/usr/bin/env python3
import tensorflow as tf
import pandas as pd
import numpy as np
import time 
import pathlib
import sys
import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, InputLayer, Conv2D, MaxPooling2D, Reshape, Dropout

tflite_models_dir = pathlib.Path("./progressive_models_lite/")
classic_models_dir = pathlib.Path('./progressive_models')

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Train and save classical TF part
# from train_convert + mnist_rpi
def FFNN(neurons, layers, epoch=10, name='ffnn'):
    model = Sequential([Flatten(input_shape=(28,28))])
    i = 0 
    while i < layers:
        model.add(Dense(neurons, activation='relu'))
        i += 1
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

    start = time.time()
    model.fit(train_images,train_labels,epochs=epoch,validation_data=(test_images, test_labels))
    end = time.time()
    process_time = round(end-start,2)

    model._name = name
    model.summary()
    print('Saved model : ./progressive_models/'+name+str(neurons)+'_'+str(layers)+'.h5')
    model.save('./progressive_models/'+name+str(neurons)+'_'+str(layers)+'.h5')

    return model, process_time

def CNN(filters=12,layers=2, epoch=10, name='cnn'):
    model = Sequential([InputLayer(input_shape=(28,28))])
    model.add(Reshape(target_shape=(28,28,1)))
    i = 0 
    while i < layers:
        model.add(Conv2D(filters,kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        i += 1
    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])

    start = time.time()
    model.fit(train_images,train_labels,epochs=epoch, validation_data=(test_images, test_labels))
    end = time.time()
    process_time = round(end-start,2)

    model._name = name
    print('Saved model : ./progressive_models/'+name+'_'+str(layers)+'.h5')
    model.save('./progressive_models/'+name+'_'+str(layers)+'.h5')
    
    return model, process_time


#[5,10,50,128,256,512,1024,1500,2048,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,
neurons = [7500,8000,8500,9000,9500,10000]
layers = [2,3,4]#,6,8,10]
dic = {} #Save the training times

if sys.argv[1] == 'ffnn':
    for n in neurons:
        dic[n] = []
        for l in layers:
            _, proc_time = FFNN(n,l,10)
            dic[n].append(proc_time)
elif sys.argv[1] == 'cnn':
    for l in layers:
        dic[l] = []
        _, proc_time = CNN(32,l,10)
        dic[l].append(proc_time)

dfres = pd.DataFrame.from_dict(dic)
dfres.index = layers
print(dfres)
#dfres.to_csv('./saved_results/fnn_inferences/h5_training_times.csv')