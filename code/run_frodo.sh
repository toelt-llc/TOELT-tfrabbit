#!/bin/bash
#TODO, short explanation

declare -i runs=3
for ((i=1; i<=${runs}; i++))
do  
    echo 'Run num '${i}' of '${runs}
    echo '_________________________'
    string="/frodo/batch3_"${i}
    python3 mnist_rpi7.py -n 5,10,50,256,512,1024,1500,2048,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000,8500,9000,9500,10000 -l 2,4,6,8 -r $string
done
echo 'Runs completed : '${runs}'/'${runs}
