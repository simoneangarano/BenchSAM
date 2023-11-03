for p in sparsegpt
do
    for s in 0
    do
        python3 main.py --cuda 2 --dataset coco --model MobileSAM --edge_filter 1 --edge_width 20 \
                        --experiment filter20_ --pruning_method $p --sparsity $s --weights mobile_sam.pt # --crop_mask 
    done
done