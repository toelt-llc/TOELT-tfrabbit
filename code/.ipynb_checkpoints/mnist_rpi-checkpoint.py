import numpy as np
import keras
import keras.datasets.mnist
import time

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from keras.layers import Dense, Flatten


(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,) 

#print("Sizes...")
#print((x_train.shape,y_train.shape))
#print((x_test.shape,y_test.shape))

# Convert y_train into one-hot format
# !!! Run only once
temp = []
for i in range(len(y_train)):
    temp.append(to_categorical(y_train[i], num_classes=10))
y_train = np.array(temp)
# Convert y_test into one-hot format
temp = []
for i in range(len(y_test)):    
    temp.append(to_categorical(y_test[i], num_classes=10))
y_test = np.array(temp)

model4 = Sequential()
model4.add(Flatten(input_shape=(28,28)))
model4.add(Dense(5, activation='sigmoid'))
model4.add(Dense(10, activation='softmax'))

model4.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['acc'])
              
start = time.time()
model4.fit(x_train, y_train, epochs=10,validation_data=(x_test,y_test), batch_size=128)
end = time.time()

print("-------------------------------------------------")
print("Benchmark Results for this test")
print()
print("Elapsed Time (min):",(end - start)/60.0)
print(end-start, "seconds")
print("-------------------------------------------------")

