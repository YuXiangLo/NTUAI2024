#!/bin/bash

# Number of parallel instances to run
NUM_THREADS=20

mkdir -p logs

# Loop to start each instance
for i in $(seq 1 $NUM_THREADS); do
    echo "Starting instance $i..."
    python test.py 2> "logs/stderr_$i.log" &
done

# Wait for all background jobs to finish
wait
echo "All instances completed."

