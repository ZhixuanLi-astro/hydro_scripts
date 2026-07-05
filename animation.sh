#!/bin/bash

names=("fig_snow_2d" "mmax" "trelax" "2ddust" "2dprop" "St")
for name in ${names[@]}; do
	echo "Creating animation for $name"
	python animation.py $name & #this & allows multiple animations to be created simultaneously
done

wait

echo "All animations have been created."
