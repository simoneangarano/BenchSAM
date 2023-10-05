#!/bin/bash

for sparsity in 0
do
    python3 step_1.py --sparsity $sparsity
done
