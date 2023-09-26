#!/bin/bash

for sparsity in 0 10 50 75 80 90
do
    python3 step_1.py --sparsity $sparsity
done
