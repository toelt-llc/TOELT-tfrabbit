#!/usr/bin/env python3
import sys, getopt
import pandas as pd
import numpy as np
import time

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras import Sequential
from keras.layers import Dense, Flatten

# Global results dictionnary, used in main() and run_model() functions 
dicres = {'Neurons':[],"Layers":[],"Training time":[], "Prediction time":[], "By image":[], 'Loss':[], 'Acc':[]}
    

# Goal : v7 but with an option on the dataset, default is mnist
# mnist_rpi8.py -n neurons -l layers -r resname -d {mnist|fashion}

def main(argv):
    """
    1. Manages the args
    neurons     : amount of neurons in the n inner layers of the current NN (n = 2)
    predictions : amount of test examples on yhich the trained model is then infered
    resultname  : name used for this specific prediction to produce the right .csv, (use the device name)

    2. Runs the data loading, model training and prediction functions

    3. Saves the result -- (Also prints result dataframe when completed)
    Under code/saved/results/ , with the given csv name
    """
    neurons_list = ''
    result = ''
    dataset = 'mnist'
    predictions = 10000
    try:
        opts, args = getopt.getopt(argv,"hn:l:r:d:")
        if len(sys.argv) == 1:
            print('! No args !')
            print('Usage : args.py -n \'neurons\' -l layers -r name -d dataset')
    except getopt.GetoptError:
        print('Usage : args.py -n \'neurons\' -l layers -r name -d dataset')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('Usage : args.py -n \'neurons\' -l layers -r name -d dataset')
            sys.exit()
        elif opt in ("-n", "--neurons"):
            neurons_list = convert(arg)
        elif opt in ("-r", "--saved_result"):
            result = arg
        elif opt in ("-l", "layers"):
            layers = convert(arg)
        elif opt == '-d':
            dataset = arg
    if len(sys.argv) > 1:
        print('Neurons array is :', neurons_list)
        print('Layers array is :', layers)
        print('Prediction is :', predictions)
    
    x_train, x_test, y_train, y_test = load_data(dataset)

    for n in neurons_list:
        for l in layers:
            run_model(n, int(l), x_train, x_test, y_train, y_test)

    print('Prediction time is over {} testing examples. '.format(predictions))

    # v8
    dfres = pd.DataFrame.from_dict(dicres)
    print('Training dataset is :', dataset)
    print('Neurons : {}, Layers : {}, Prediction : {}, Result file : {}'.format(neurons_list, layers, predictions, result))
    print('Saved dataframe :', dfres)
    dfres.to_csv('./saved_results/'+ result + '.csv', index=False)

    
def convert(string):
    """
    Converts and returns the neurons string args into a list 
    ex : '5,10,15' -> [5,10,15]
    """
    li = list(string.split(","))
    return li

def load_data(data):
    """
    Loads the dataset : classic fashion_mnist from Keras, checks if shape is as expected
    Converts categories into numbers from 0 to 9
    Returns x_train, x_test, y_train, y_test

    # TODO use from sklearn.preprocessing import OneHotEncoder
    https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if data == 'fashion':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    assert x_train.shape == (60000, 28, 28)
    assert x_test.shape == (10000, 28, 28)
    assert y_train.shape == (60000,)
    assert y_test.shape == (10000,) 

    # This part may need an update for fashion_mnist
    temp = []
    for i in range(len(y_train)):
        temp.append(to_categorical(y_train[i], num_classes=10))
    y_train = np.array(temp)
    temp = []
    for i in range(len(y_test)):    
        temp.append(to_categorical(y_test[i], num_classes=10))
    y_test = np.array(temp)

    # Normalization v7, the load.data() returns uint8 arrays
    x_train = x_train.copy()
    x_test = x_test.copy()
    x_train = x_train/255
    x_test = x_test/255
    
    return x_train, x_test, y_train, y_test

def run_model(n, l, x_train, x_test, y_train, y_test):
    """
    #TODO: use a real case model, cnn ? ; add the epochs and batch_size parameters 
    Args:
        n : the number of neurons, specified by the user
        l : the number of layers
        datasets : used by .fit method to train the network
        
    Sequential model with Optimizer : Adam . Loss function : MeanSquaredError
    epochs = 10, batch_size=128

    Processes & adds the training time for the result dataframe 
    """
    # Model creation 
    model = Sequential([Flatten(input_shape=(28,28))])
    i = 0 
    while i < l:
        model.add(Dense(n, activation='relu'))
        i += 1
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='MeanSquaredError', 
              optimizer='sgd',
              metrics=['acc'])
    
    # Model fit, timed
    print("===== Step : ", n, '=====')
    start = time.time()
    model.fit(x_train, y_train, epochs=10,validation_data=(x_test,y_test), batch_size=128)
    end = time.time()
    
    training_time = round(end-start, 2)
    
    #v6 + 7
    
    size = 10000
    test_sample = x_test[:size]
    #test_sample = y_test[:size]

    # Timed inference
    start1 = time.time()
    model.predict(test_sample)
    end1 = time.time()
    img_time = round((end1-start1)/size, 4 ) # Inference time for each image

    # Eval
    print('Evaluation .....')
    eval = model.evaluate(x_test, y_test)

    # From v6, not necessary but more readable
    Train = training_time
    Pred = round(end1-start1, 2)
    Img = img_time
    Loss = round(eval[0],2)
    acc = round(eval[1],2)
    
    # Dicres is the new way to save results, then turned into a pd.Df for further save(and display).
    dicres['Neurons'].append(n)
    dicres['Layers'].append(l)
    dicres['Training time'].append(Train)
    dicres['Prediction time'].append(Pred)
    dicres['By image'].append(Img)
    dicres['Loss'].append(Loss)
    dicres['Acc'].append(acc)

if __name__ == "__main__":
   main(sys.argv[1:])