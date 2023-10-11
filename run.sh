for sparsity in 50 60 70 80 90
do
    python3 main.py --cuda 2 --dataset sa1b --model SAM --edge_filter 1 \
                    --crop_mask 1 --sparsity $sparsity --experiment filter_ --pruning_method l1norm
done