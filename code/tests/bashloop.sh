#!/bin/bash
declare -i runs=2
for ((i=1; i<= ${runs}; i++))
do  
    echo 'Run num '${i}' of '${runs}
    string = "mactodel"${i}
    python3 mnist_rpi6.py -n 5,10 -l 2,3 -r string
    echo 'Runs completed : '${runs}'/'${runs}
done