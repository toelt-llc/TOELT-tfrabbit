#!/bin/bash

MODEL=./tlite/test.tflite
if [ ! -f "$FILE" ]; then
    echo "$MODEL does not exists. Attempting to train the model : "
    python3 ./tflite/trainmodel.py 
fi
python3 ./tflite/readmodel.py 
