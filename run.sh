for p in sparsegpt
do
    for s in 0
    do
        python3 main.py --cuda 1 --dataset coco --model MobileSAM --edge_filter 1 --edge_width 20 \
                        --experiment filter20_kd_ --pruning_method $p --sparsity $s --weights distilled_mobile_sam.pt # --crop_mask 
    done
done