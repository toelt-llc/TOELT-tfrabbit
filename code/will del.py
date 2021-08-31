import tensorflow as tf

from tensorflow.keras import Sequential
from keras.layers import Dense, Flatten

n = 10

model = Sequential(
    [Flatten(input_shape=(28,28))],
    [Dense(n, activation='relu')],
    [Dense(n, activation='relu')],
    [Dense(10, activation='softmax')]
    )