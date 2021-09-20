#!/bin/bash

MODEL=./tflite/test.tflite
#echo $MODEL
if [ ! -f "$MODEL" ]; then
    echo "$MODEL does not exists. Attempting to train and convert the model : "
    python3 ./tflite/trainmodel.py 
fi
echo "Running the quantized model"
python3 ./tflite/readmodel.py 
