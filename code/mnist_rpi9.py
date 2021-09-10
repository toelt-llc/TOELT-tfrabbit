#!/usr/bin/env python3
import sys, getopt
import tensorflow as tf
import pandas as pd
import numpy as np
import time

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras import Sequential
from keras.layers import Dense, Flatten

# Global results dictionnary, used in main() and run_model() functions 
dicres = {'Neurons':[],"Layers":[],"Training time":[], "Inference time":[], "By image":[], 'Loss':[], 'Acc':[],
         'Conversion time':[], 'Interpret time':[], 'Lite Inf time':[]}
    
# Goal : v8 but the model is also converterd and ran as tflite quantized
# - keep in mind, float16 option

def main(argv):
    """
    1. Manages the args
    neurons,layers     : amount of neurons in the l inner layers of the current NN 
    dataset            : on which dataset the network is trained, default mnist
    epochs, batch_size : model.fit parameters
    predictions        : amount of test examples on which the trained model is then infered
    resultname         : name used for this specific prediction to produce the right .csv, (use the device name)

    2. Runs the data loading, model training and prediction functions

    3. Saves the result -- (Also prints result dataframe when completed)
    Under code/saved_results/ , with the given csv name
    """
    neurons_list = [5,10]
    result = 'unknown'  # if no name is specified
    layers = [2]
    dataset = 'mnist'
    epochs = 10
    batch_size = 128
    predictions = 10000 # ie. nb of inference examples

    usage = 'Usage : mnist_rpi8.py -d {mnist|fashion} -n \'neurons\' -l layers  -e epochs -b batch_size -r resultname'
    try:
        opts, args = getopt.getopt(argv,"hn:l:r:d:e:b:")
        if len(sys.argv) == 1:
            print('! No args !')
            print(usage)
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(usage)
            sys.exit()
        elif opt in ("-n", "--neurons"):
            neurons_list = convert(arg)
        elif opt in ("-r", "--saved_result"):
            result = arg
        elif opt in ("-l", "layers"):
            layers = convert(arg)
        elif opt == '-d':
            dataset = arg
        elif opt == '-e':
            epochs = int(arg)
        elif opt == '-b':
            batch_size = int(arg)
    if len(sys.argv) > 1:
        print('Dataset :', dataset)
        print('Neurons array :', neurons_list)
        print('Layers array :', layers)
        print('Prediction :', predictions)
        print('Epochs :', epochs)
        print('Batch size :', batch_size)
    
    x_train, x_test, y_train, y_test = load_data(dataset)

    for n in neurons_list:
        for l in layers:
            mod = run_model(n, int(l), x_train, x_test, y_train, y_test, epochs, batch_size)
            run_tflite(mod, dataset)

    # v8
    # dfres is the results from classical tf
    dfres = pd.DataFrame.from_dict(dicres)
    print('Training dataset is :', dataset)
    print('Neurons : {}, Layers : {}, Epochs : {}, Batch_size : {}, Inference exs : {}, Result file : {}'
        .format(neurons_list, layers, epochs, batch_size, predictions, result))
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

    # from sklearn.preprocessing import OneHotEncoder
    https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if data == 'fashion':
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    assert x_train.shape == (60000, 28, 28) # only valuable for mnists datasets
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
    x_train = x_train.astype(np.float32)/255
    x_test = x_test.astype(np.float32)/255
    
    return x_train, x_test, y_train, y_test

def run_model(n, l, x_train, x_test, y_train, y_test, epochs, batch_size):
    """
    #TODO: use a real case model, cnn ? ; add the epochs and batch_size parameters 
            add new args

    Creates, trains, and runs a model
    Args:
        n : the number of neurons, specified by the user, default (5,10)
        l : the number of layers, default (2)
        datasets : used by .fit method to train the network, default (mnist)
        epochs & batch_size : model.fit parameters, default (10,128)
        
    Sequential model with Optimizer : Adam . Loss function : MeanSquaredError

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
    model.fit(x_train, y_train, epochs=epochs,validation_data=(x_test,y_test), batch_size=batch_size)
    end = time.time()
    training_time = round(end-start, 2)
    
    # Timed inference
    size = 10000
    test_sample = x_test[:size]
    start_test = time.time()
    model.predict(test_sample)
    end_test = time.time()
    inf_time = round(end_test-start_test, 2)        # Total inference time
    inf_img = round((end_test-start_test)/size, 4 ) # Inference time for each image

    # Model eval
    print('Evaluation .....')
    eval = model.evaluate(x_test, y_test)
    loss = round(eval[0],2)
    acc = round(eval[1],2)
    
    # Dicres is the new way to save results, then turned into a pd.Df for further save(and display).
    dicres['Neurons'].append(n)
    dicres['Layers'].append(l)
    dicres['Training time'].append(training_time)
    dicres['Inference time'].append(inf_time)
    dicres['By image'].append(inf_img)
    dicres['Loss'].append(loss)
    dicres['Acc'].append(acc)

    return model


def run_tflite(tf_model, data):
    x_train, x_test, y_train, y_test = load_data(data)
    print(x_train.dtype)
    start_conv = time.time()
    # TODO set option to read it from saved model or from existing model
    # for now : uses keras (existing model)
    #model_path = './tflite/model2_mnist/'
    #converter1 = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter1 = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    def representative_data_gen():
        for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(1000):
            #input_value = input_value.astype(np.float32)
            yield [input_value]
    converter1.representative_dataset = representative_data_gen
    converter1.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter1.optimizations = [tf.lite.Optimize.DEFAULT]
    converter1.inference_input_type = tf.uint8
    converter1.inference_output_type = tf.uint8

    quant_model = converter1.convert()
    end_conv = time.time()
    dicres['Conversion time'].append(end_conv-start_conv)

    start_int = time.time()
    interpreter = tf.lite.Interpreter(model_content=quant_model)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_type = interpreter.get_input_details()[0]['dtype']
    # Quantization parameters : 
    input_scale, input_zero_point = input_details[0]["quantization"]

    # whole set, currently only supports test_set as an array
    print(np.shape(y_train))
    print(interpreter.get_output_details())
    #interpreter.allocate_tensors()
    interpreter.resize_tensor_input(input_index=input_details[0]['index'], tensor_size=(np.shape(y_train)))
    interpreter.allocate_tensors()

    # 8bit quantization approximation
    test_images_q = y_train / input_scale + input_zero_point
    test_images_q = np.reshape(test_images_q.astype(input_type), np.shape(y_train)) # wordy line

    # Loading into the tensor
    interpreter.set_tensor(input_details[0]['index'], test_images_q)
    end_int = time.time()

    dicres['Interpret time'].append(end_int-start_int)
    
    output_details = interpreter.get_output_details()
    start_inf = time.time()
    interpreter.invoke()
    end_inf = time.time()

    inference = interpreter.get_tensor(output_details[0]['index'])
    predictions = np.argmax(inference, axis=1)

    dicres['Lite Inf time'].append(end_inf-start_inf)
    print('Quantized model accuracy : ', (predictions == y_test).mean())

    return predictions


if __name__ == "__main__":
   main(sys.argv[1:])