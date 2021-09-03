#!/bin/bash
declare -i runs=1
for ((i=1; i<=${runs}; i++))
do  
    echo 'Run num '${i}' of '${runs}
    echo '_________________________'
    string="pi/longpi"${i}
    python3 mnist_rpi5.py -n 7000 -r $string
done
echo 'Runs completed : '${runs}'/'${runs}