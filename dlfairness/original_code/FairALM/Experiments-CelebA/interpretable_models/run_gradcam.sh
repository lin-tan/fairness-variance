#!/bin/bash

models=( 'N' 'G' 'F' )
image_name=( '162821' '162787' '162803' '162834' )  # Main 
#image_name=( '173141' '173453' '173556' '173638' '173789' '173899' '174056' '174113' )  # Supp male
#image_name=( '162874' '162878' '162887' '162906' '162925' '162929' '162955' '162989' ) # Supp female

for i in "${image_name[@]}"
do
    for j in "${models[@]}"
    do
	echo $i $j
	python gradcam.py --imname $i --modelname $j
    done
done
