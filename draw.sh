#! /bin/bash

# for i in {900..3800}; do
# 	echo "Processing item $i"
# 	python plot.py $i ./new_snowline
# done
# # python animation.py

# Number of parallel processes (adjust based on your CPU cores)
NUM_PROCESSES=40

# Define the range of values
START=80
END=117

# Function to run the plotting script
run_plot() {
	for ((i = $1; i <= $2; i++)); do
		echo "Processing item $i"
		python plot.py $i passive_test &
		# Limit the number of background processes
		if (($(jobs -r | wc -l) >= NUM_PROCESSES)); then
			wait -n
		fi
	done
	wait
}

# Split the range into chunks for parallel processing
STEP=$((($END - $START + 1) / $NUM_PROCESSES))
for ((p = 0; p < $NUM_PROCESSES; p++)); do
	CHUNK_START=$((START + p * STEP))
	CHUNK_END=$((CHUNK_START + STEP - 1))
	if ((p == NUM_PROCESSES - 1)); then
		CHUNK_END=$END
	fi
	run_plot $CHUNK_START $CHUNK_END &
done

# Wait for all background processes to finish
wait
