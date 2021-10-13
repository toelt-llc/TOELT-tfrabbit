#!/usr/bin/env python3
import tensorflow as tf
import pandas as pd
import numpy as np
import time 
import pathlib
import pickle
import sys
import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

classic_models_dir = pathlib.Path('./progressive_models')

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

# Run all kept h5 models, to have an idea on accuracy changes.

dic = {} #Save the converting times
for _,_,models in os.walk('./progressive_models/'):
    for model in sorted(models):
        dic[model] = []
        start = time.time()
        loaded_model = tf.keras.models.load_model('./progressive_models/'+model)
        loss,acc = loaded_model.evaluate(test_images, test_labels)
        # loss = round(eval[0],2)
        # acc = round(eval[1],2)
        end = time.time()

        dic[model].append(acc)
        #dic[model].append(round(end-start,2))

dfres = pd.DataFrame.from_dict(dic)
print(dfres)
dfres.to_csv('./saved_results/fnn_inferences/accuracy_h5.csv')
with open('./saved_results/fnn_inferences/accuracy_dict.pkl', 'wb') as f:
    pickle.dump(dic, f)
