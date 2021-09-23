import tensorflow as tf
import numpy as np

cifar = tf.keras.datasets.cifar100
(train_images, train_labels), (test_images, test_labels) = cifar.load_data()
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0


print(test_images.shape)
print((test_labels.shape))

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(test_images.shape)