#!/usr/bin/python
import sys, getopt
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import time

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras import Sequential
from keras.layers import Dense, Flatten


dfres =  pd.DataFrame( columns=["Execution time ", "Prediction time", " By image"])
exec_times = []
pred_times_tot = []
pred_times1 = []
    
# Version that saves the dfres    
def main(argv):
    """
    1. Manages the args
    neurons     : amount of neurons in the n inner layers of the current NN (n = 2)
    predictions : amount of test examples on yhich the trained model is then infered
    resultname  : name used for this specific prediction to produce the right .csv, (use the device name)

    2. Runs the data loading, model training and prediction functions

    3. Saves the result
    Under code/saved/results/ , with the given csv name
    
    Also prints result dataframe when completed

    """
    neurons_list = ''
    predictions = ''
    result = ''
    try:
        opts, args = getopt.getopt(argv,"hn:p:r:",["neurons=","predictions=","saved_result="])
        if len(sys.argv) == 1:
            print('! No args !')
            print('Usage : args.py -n \'neurons\' -p <npredictions> -r resultname')
    except getopt.GetoptError:
        print('Usage : args.py -n \'neurons\' -p <npredictions> -r resultname')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage : args.py -n \'neurons\' p <npredictions> -r resultname')
            sys.exit()
        elif opt in ("-n", "--neurons"):
            arr = sys.argv[1].split(',')
            neurons_list = convert(arg)
        elif opt in ("-p", "--npredictions"):
            predictions = int(arg)
        elif opt in ("-r", "--saved_result"):
            result = arg

    if len(sys.argv) > 1:
        print('Neurons array is :', neurons_list)
        print('Prediction is :', predictions)
    
    x_train, x_test, y_train, y_test = loading()
    
    for n in neurons_list:
        run_model(n, x_train, x_test, y_train, y_test)
        predict_time(n, predictions, x_train, y_train)

    dfres.index.name = 'Neurons'
    dfres.to_csv('./saved_results/'+ result + '.csv')
    #dfres.to_pickle('./saved_results/' + result + '.pkl')
    print(dfres)
    print('Prediction time is over {} testing examples. '.format(predictions))


    
def convert(string):
    """
    Converts and returns the neurons string args into a list 
    ex : '5,10,15' -> [5,10,15]
    """
    li = list(string.split(","))
    return li

def loading():
    """
    Loads the classic mnist dataset from Keras, check if shape is as expected
    Converts categories into numbers from 0 to 9
    Returns x_train, x_test, y_train, y_test

    # TODO use from sklearn.preprocessing import OneHotEncoder
    https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,) 

    temp = []
    for i in range(len(y_train)):
        temp.append(to_categorical(y_train[i], num_classes=10))
    y_train = np.array(temp)
    temp = []
    for i in range(len(y_test)):    
        temp.append(to_categorical(y_test[i], num_classes=10))
    y_test = np.array(temp)
    
    return x_train, x_test, y_train, y_test

def run_model(n, x_train, x_test, y_train, y_test):
    """
    Args:
        n : the number of neurons, specified by the user
        datasets : used by .fit method to train the network
    Returns:
        none, but saves the model in /tflite/bench_model/ for further use

    Sequential model with 1 input layer, 2 inner layers and 1 output layer. 
    Optimizer : Adam . Loss function : MeanSquaredError

    Processes & adds the training time to the result df 
    """
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(n, activation='relu'))
    model.add(Dense(n, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.compile(loss='MeanSquaredError', 
              optimizer='adam',
              metrics=['acc'])
    
    print("===== Step : ", n, '=====')
    start = time.time()
    model.fit(x_train, y_train, epochs=10,validation_data=(x_test,y_test), batch_size=128)
    end = time.time()
    
    exec_times.append(round(end-start, 2))
    dfres.loc[n] = round(end-start, 2)
    
    model.save('./tflite/bench_model')
    
def predict_time(n, size, x_test, y_test):
    """
    Args:
        n    : the number of neurons, specified by the user
        size : size of the test set prediction, max is 10000

    Processes & adds the inference time on the tests to the result df + computes time/img 
    # TODO output accuracy
    """
    model = keras.models.load_model('./tflite/bench_model')
    train_sample = x_test[:size]
    test_sample = y_test[:size]

    start1 = time.time()
    preds = model.predict(train_sample)
    end1 = time.time()
    
    img_time = round((end1-start1)/size, 4 )
    pred_times_tot.append(end1-start1)
    pred_times1.append(img_time)
    
    dfres.loc[n]['Prediction time'] = round(end1-start1, 2)
    dfres.loc[n][2] = img_time

if __name__ == "__main__":
   main(sys.argv[1:])