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
exec_times = []

pred_times_tot = []
pred_times1 = []

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
    exec_times.append(round(end-start, 2))
    
    model.save('./bench_model')
    
def predict_time(size):
    model = keras.models.load_model('./bench_model')
    train_sample = x_train[:size]
    test_sample = y_train[:size]

    start1 = time.time()
    preds = model.predict(train_sample)
    end1 = time.time()
    
    img_time = round((end1-start1)/size, 4 )
    print('Time to classify ', size, ' images : ', end1-start1)
    print('Average time to classify 1 image : ', img_time)
    pred_times_tot.append(end-start)
    pred_times1.append(img_time)

for n in neurons:
    run_model(n)
    predict_time(50000)

res = zip(neurons, exec_times, pred_times_tot)
print(list(res))
