#!/bin/bash
for ((i=1; i<=10; i++))
do  
    string = "frodo"${i}
    python3 mnist_rpi5.py -n 5,10,50,128,256,512,1024,1500,2048,2500,3000,3250,3500,3750,4096,4500 -p 10000 -r string
done