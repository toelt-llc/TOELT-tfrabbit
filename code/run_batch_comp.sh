#!/bin/bash

#This loops starts at 0 
declare -i runs=10
for ((i=0; i<=${runs}; i++))
do  
    echo 'Run num '${i}' of '${runs}
    echo '_________________________'
    string="/frodo/batch_comp"${i} 
    #check the mnist_rpi version
    python3 mnist_rpi8.py -n 128,256 -l 2 -r $string -b $(( 2**i )) 
done
echo 'Runs completed : '${runs}'/'${runs}
