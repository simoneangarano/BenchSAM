for sparsity in 10 20 30 40 50 60 70 80
do
    python3 main.py --cuda 1 --dataset sa1b --model SAM --edge_filter 1 \
                    --crop_mask 1 --sparsity $sparsity --experiment filter_ --pruning_method l1norm
done
