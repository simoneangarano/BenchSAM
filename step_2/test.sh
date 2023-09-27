#!/bin/bash

for sparsity in 90
do
    python3 step_2.py --sparsity $sparsity
done
