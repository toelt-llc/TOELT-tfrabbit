#!/bin/bash
declare -i runs=1
for ((i=1; i<=${runs}; i++))
do  
    echo 'Run num '${i}' of '${runs}
    echo '_________________________'
    string="/frodo/single"${i}
    python3 mnist_rpi5.py -n 10000  -r $string
done
echo 'Runs completed : '${runs}'/'${runs}