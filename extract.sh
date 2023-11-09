for p in sparsegpt
do
    for s in 0
    do
        python3 main.py --cuda 3 --dataset coco --model MobileSAM --edge_filter 1 --edge_width 20 \
                        --experiment filter20_ --pruning_method $p --sparsity $s \
                        --weights distilled_mobile_sam_decoder_fd_l_2.pt --suffix _fd # --crop_mask 
    done
done