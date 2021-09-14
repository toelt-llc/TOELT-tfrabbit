from numpy.lib.twodim_base import tri
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time 

def load_data():
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print(train_images.dtype)

    # Normalize the input image so that each pixel value is between 0 to 1.
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    return train_images, train_labels, test_images, test_labels

def trained_model():
    train_images, train_labels, test_images, test_labels = load_data()
    model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28, 28)),
    tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=True),
                metrics=['accuracy'])
    model.fit(train_images,train_labels,epochs=1,validation_data=(test_images, test_labels))
    

    # model = trained_model()
    return model


####  TFlite conv + inference 
inf_time = []
train_images, train_labels, test_images, test_labels = load_data()
def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(1000):
        yield [input_value]

def conv_int8(model_path):
    start_conv = time.time()
    # TODO set option to read it from saved model or from existing model
    # for now : uses keras (existing model)
    # converter1 = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter1 = tf.lite.TFLiteConverter.from_keras_model(model_path)
    converter1.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter1.optimizations = [tf.lite.Optimize.DEFAULT]
    converter1.representative_dataset = representative_data_gen
    converter1.inference_input_type = tf.uint8
    converter1.inference_output_type = tf.uint8

    quant_model = converter1.convert()
    end_conv = time.time()
    inf_time.append(end_conv-start_conv)

    return(quant_model)


def interpret(model, test_set):
    #start_int = time.time()
    interpreter = tf.lite.Interpreter(model_content=model)

    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    input_type = interpreter.get_input_details()[0]['dtype']
    # Quantization parameters : 
    input_scale, input_zero_point = input_details[0]["quantization"]

    # whole set, currently only supports test_set as an array
    interpreter.resize_tensor_input(input_index=input_details[0]['index'], tensor_size=np.shape(test_set))
    interpreter.allocate_tensors()

    # 8bit quantization approximation
    test_images_q = test_set / input_scale + input_zero_point
    test_images_q = np.reshape(test_images_q.astype(input_type), np.shape(test_set)) # wordy line

    # Loading into the tensor
    interpreter.set_tensor(input_details[0]['index'], test_images_q)
    end_int = time.time()

    #inf_time.append(end_int-start_int)
    return interpreter

def run_inference(interpreter):
    output_details = interpreter.get_output_details()
    start_inf = time.time()
    interpreter.invoke()
    end_inf = time.time()
    inference = interpreter.get_tensor(output_details[0]['index'])
    predictions = np.argmax(inference, axis=1)

    inf_time.append(end_inf-start_inf)

    return predictions

model = trained_model()
quanted_model = conv_int8(model)
interpreter = interpret(quanted_model, test_images)

_, eval = model.evaluate(test_images, test_labels)
start = time.time()
preds = model.predict(test_images)
dec = np.argmax(preds, axis=1)
end = time.time()
inftime = end-start
print("Classic TF test acc:", round(eval, 3))
print("Classic TF inference time:", round(inftime, 3))

predictions = run_inference(interpreter)
print('Quantized model acc : ', (predictions == test_labels).mean())
print('Conversion time : {}, Inference time {}'.format(round(inf_time[0],2), round(inf_time[1],2)))

open("test.tflite", "wb").write(quanted_model)
#tf.saved_model.save(model, './model3_mnist/')
#model.save('./model3_keras/')