#!/bin/bash
declare -i runs=2
for ((i=1; i<=${runs}; i++))
do  
    echo 'Run num '${i}' of '${runs}
    echo '_________________________'
    string="pi/longpi"${i}
    python3 mnist_rpi5.py -n 5,10,50,128,256,512,1024,1500,2048,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000 -r $string
done
echo 'Runs completed : '${runs}'/'${runs}