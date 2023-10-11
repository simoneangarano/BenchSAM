for sparsity in 60 70 80
do
    python3 main.py --cuda 5 --dataset coco --model SAM --edge_filter 1 \
                    --sparsity $sparsity --experiment filter_ --pruning_method sparsegpt
done