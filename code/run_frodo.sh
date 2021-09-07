#!/bin/bash
declare -i runs=3
for ((i=1; i<=${runs}; i++))
do  
    echo 'Run num '${i}' of '${runs}
    echo '_________________________'
    string="/frodo/100k"${i}
    python3 mnist_rpi5.py -n 10000,20000,30000,40000 -r $string
done
echo 'Runs completed : '${runs}'/'${runs}