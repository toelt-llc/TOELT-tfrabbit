#!/bin/bash
declare -i runs=3
for ((i=1; i<= ${runs}; i++))
do  
    echo 'Run num '${i}' of '${runs}
    echo '_________________________'
    string="mactodel"${i}
    python3 mnist_rpi6.py -n 5,10,15 -l 5,6,7 -r $string
    #echo $string
done
echo 'Runs completed : '${runs}'/'${runs}