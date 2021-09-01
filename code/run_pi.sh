#!/bin/bash
declare -i runs=10
for ((i=1; i<=${runs}; i++))
do  
    echo 'Run num '${i}' of '${runs}
    echo '_________________________'
    string = "pi"${i}
    python3 mnist_rpi6.py -n 5,10,50,128,256,512,1024,1500,2048,2500,3000,3250,3500,3750,4096,4500 -r $string
done
echo 'Runs completed : '${runs}'/'${runs}