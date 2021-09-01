#!/bin/bash
declare -i runs=2
for ((i=1; i<= ${runs}; i++))
do  
    echo 'Run num '${i}' of '${runs}
    echo '_________________________'
    string="mactodel"${i}
    python3 mnist_rpi6.py -n 5 -l 5 -r $string
    #echo $string
done
echo 'Runs completed : '${runs}'/'${runs}