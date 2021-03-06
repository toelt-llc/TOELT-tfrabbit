import tensorflow as tf
import tensorflow_datasets as tfds
import time

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

def create_model(n):
    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(n,activation='relu'),
      tf.keras.layers.Dense(n,activation='relu'),
      tf.keras.layers.Dense(n,activation='relu'),
      tf.keras.layers.Dense(n,activation='relu'),
      tf.keras.layers.Dense(n,activation='relu'),
      tf.keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    return model

#####################################################

times_list_GPU = []
times_list_noGPU = []
neuron_list = [1,10,100,500, 1000, 1500, 2000, 5000, 10000]

with tf.device('/CPU:0'):
    for i in neuron_list:
        print("-------- Number of neurons:",i)
        model = create_model(i)

        start = time.time()
        model.fit(
            ds_train,
            epochs=1,
            validation_data=ds_test,
        )
        end = time.time()
        print("Elapsed Time (min):",(end - start)/60.0)
        print("-------------------------------------------------")
        times_list_noGPU.append((end - start)/60.0)

with tf.device('/GPU:0'):
    for i in neuron_list:
        print("-------- Number of neurons:",i)
        model = create_model(i)

        start = time.time()
        model.fit(
            ds_train,
            epochs=1,
            validation_data=ds_test,
        )
        end = time.time()
        print("Elapsed Time (min):",(end - start)/60.0)
        print("-------------------------------------------------")
        times_list_GPU.append((end - start)/60.0)

print("Times with GPU:")
print(times_list_GPU)
print("Times without GPU:")
print(times_list_noGPU)
print("Neurons:")
print(neuron_list)
