from tensorflow.keras.datasets import cifar10
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.utils.multiclass import unique_labels
import os
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# Keras import
from tensorflow.keras import Sequential
from tensorflow.keras.applications import VGG19,ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import Flatten,Dense,BatchNormalization,Activation,Dropout
from tensorflow.keras.utils import to_categorical



# We load the CIFAR-10 dataset
(x_train,y_train),(x_test,y_test)=cifar10.load_data()

print("Sizes...")
print((x_train.shape,y_train.shape))
print((x_test.shape,y_test.shape))

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

# Data Generators
train_generator = ImageDataGenerator(
                                    rotation_range=2,
                                    horizontal_flip=True,
                                    zoom_range=.1 )

test_generator = ImageDataGenerator(
                                    rotation_range=2,
                                    horizontal_flip= True,
                                    zoom_range=.1)
train_generator.fit(x_train)
test_generator.fit(x_test)

# Learning rate decay
lrr= ReduceLROnPlateau(
                       monitor='val_acc', #Metric to be measured
                       factor=.01, #Factor by which learning rate will be reduced
                       patience=3,  #No. of epochs after which if there is no improvement in the val_acc, the learning rate is reduced
                       min_lr=1e-5) #The minimum learning rate


base_model_1 = VGG19(include_top=True,weights=None,input_shape=(32,32,3),classes=y_train.shape[1])
base_model_2 = ResNet50(include_top=True,weights=None,input_shape=(32,32,3),classes=y_train.shape[1])

print(base_model_1.summary())

batch_size = 100
epochs = 1
learn_rate = 0.01

adam=Adam(learning_rate=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

base_model_1.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
base_model_1.fit(train_generator.flow(x_train,y_train,batch_size=batch_size),
                      epochs=epochs,
                      steps_per_epoch=x_train.shape[0]//batch_size,
                      validation_data=test_generator,validation_steps=250,
                      callbacks=[lrr],verbose=1)
