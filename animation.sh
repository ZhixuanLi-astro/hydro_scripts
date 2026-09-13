#!/bin/bash

names=("fig_snow_2d" "2ddust")
for name in ${names[@]}; do
	echo "Creating animation for $name"
	python animation.py $name &
	allows multiple animations to be created simultaneously
done

wait

echo "All animations have been created."
