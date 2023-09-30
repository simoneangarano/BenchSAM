#!/bin/bash

for sparsity in 50
do
    python3 step_2.py --sparsity $sparsity --center_prompt False --experiment thr_ --cuda 2
done
