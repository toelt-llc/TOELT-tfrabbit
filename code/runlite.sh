#!/bin/bash

python3 ./tflite/trainmodel.py -d data
python3 ./tflite/readmodel.py -d data -m model
