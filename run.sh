for p in sparsegpt
do
    for s in 90
    do
        python3 main.py --cuda 3 --dataset coco --model SAM --edge_filter 1 --edge_width 20 \
                        --experiment filter20_ --pruning_method $p --sparsity $s # --crop_mask 
    done
done