#!/bin/bash

names=("fig_snow_2d" "2ddust" "2ddust_comp" "2ddust_rho")
for name in ${names[@]}; do
	echo "Creating animation for $name"
	python animation.py $name apng &
	allows multiple animations to be created simultaneously
done

wait

echo "All animations have been created."
