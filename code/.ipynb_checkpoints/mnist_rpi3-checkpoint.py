#!/usr/bin/python
import sys, getopt
import numpy as np
import tensorflow as tf
import keras
import time
import pandas as pd

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from keras.layers import Dense, Flatten


dfres =  pd.DataFrame( columns=["Execution time ", "Prediction time", " By image"]) 
exec_times = []
pred_times_tot = []
pred_times1 = []
    
    
def main(argv):
    neurons_list = ''
    predictions = ''
    try:
        opts, args = getopt.getopt(argv,"hn:p:",["neurons=","predictions="])
        if len(sys.argv) == 1:
            print('! No args !')
            print('Usage : args.py -n \'neurons\' p <npredictions>')
            print('Default is n = 5,10,100,500 and p = 50000')
            #neurons_list = '5,10,100,500'
            #predictions = 50000
    except getopt.GetoptError:
        print('Usage : args.py -n \'neurons\' p <npredictions>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage : args.py -n \'neurons\' p <npredictions>')
            print('Default neurons array is [5, 10, 100, 500], predictions to compute is 50000')
            sys.exit()
        elif opt in ("-n", "--neurons"):
            arr = sys.argv[1].split(',')
            neurons_list = convert(arg)
        elif opt in ("-p", "--npredictions"):
            predictions = int(arg)

    if len(sys.argv) > 1:
        print('Neurons array is :', neurons_list)
        print('Prediction is :', predictions)
    
    x_train, x_test, y_train, y_test = loading()
    
    for n in neurons_list:
        run_model(n, x_train, x_test, y_train, y_test)
        predict_time(n, predictions, x_train, y_train)

    print(dfres)
    print('Prediction time is over {} training examples. '.format(predictions))


    
def convert(string):
    li = list(string.split(","))
    return li

def loading():
    # mnist check
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,) 

    # Convert y_train into one-hot format
    temp = []
    for i in range(len(y_train)):
        temp.append(to_categorical(y_train[i], num_classes=10))
    y_train = np.array(temp)
    # Same for y_test
    temp = []
    for i in range(len(y_test)):    
        temp.append(to_categorical(y_test[i], num_classes=10))
    y_test = np.array(temp)
    
    return x_train, x_test, y_train, y_test

def run_model(n, x_train, x_test, y_train, y_test):
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(n, activation='relu'))
    model.add(Dense(n, activation='relu'))
    model.add(tf.keras.layers.Lambda(lambda x: tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)))
    model.add(Dense(10, activation='relu'))
    model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['acc'])
    
    print("===== Step : ", n, '=====')
    start = time.time()
    model.fit(x_train, y_train, epochs=10,validation_data=(x_test,y_test), batch_size=128)
    end = time.time()
    
    exec_times.append(round(end-start, 2))
    dfres.loc[n] = round(end-start, 2)
    
    model.save('./tflite/bench_model')
    
def predict_time(n, size, x_train, y_train):
    model = keras.models.load_model('./tflite/bench_model')
    train_sample = x_train[:size]
    test_sample = y_train[:size]

    start1 = time.time()
    preds = model.predict(train_sample)
    end1 = time.time()
    
    img_time = round((end1-start1)/size, 4 )
    pred_times_tot.append(end1-start1)
    pred_times1.append(img_time)
    
    dfres.loc[n]['Prediction time'] = round(end1-start1, 2)
    dfres.loc[n][2] = img_time
    #print('Time to classify ', size, ' images : ', end1-start1)
    #print('Average time to classify 1 image : ', img_time)

if __name__ == "__main__":
   main(sys.argv[1:])