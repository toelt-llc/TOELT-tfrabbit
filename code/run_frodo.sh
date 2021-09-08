#!/bin/bash
#TODO, short explanation

declare -i runs=1
for ((i=1; i<=${runs}; i++))
do  
    echo 'Run num '${i}' of '${runs}
    echo '_________________________'
    string="/frodo/mnist7_test"${i}
    python3 mnist_rpi7.py -n 250 -l 2 -r $string
done
echo 'Runs completed : '${runs}'/'${runs}