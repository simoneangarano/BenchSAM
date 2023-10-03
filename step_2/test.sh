#!/bin/bash

for sparsity in 0
do
    python3 step_2.py --sparsity $sparsity --cuda 0
done
