import numpy as np
import keras
import time

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from keras.layers import Dense, Flatten


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


neurons = [5, 10, 100]#, 500, 1000]#, 5000, 10000]#, 100000]
times = []

def run_model(n):
    model = Sequential()
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(n, activation='relu'))
    model.add(Dense(n, activation='relu'))
    model.add(Dense(10, activation='relu'))
    
    model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['acc'])
    
    print("===== Step : ", n, '=====')
    start = time.time()
    model.fit(x_train, y_train, epochs=10,validation_data=(x_test,y_test), batch_size=128)
    end = time.time()
    times.append(round(end-start, 2))

for n in neurons:
    run_model(n)

res = zip(neurons, times)
print(list(res))
