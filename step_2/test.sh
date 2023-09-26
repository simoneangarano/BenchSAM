#!/bin/bash

for sparsity in 10 50 75 80 90
do
    python3 step_2.py --sparsity $sparsity
done
