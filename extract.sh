for p in sparsegpt
do
    for s in 0
    do
        python3 main.py --cuda 7 --dataset sa1b --model MobileSAM --edge_filter 1 --edge_width 20 \
                        --experiment filter20_ --pruning_method $p --sparsity $s --suffix _kd \
                        --weights distilled_mobile_sam_decoder_distill_16.pt --rle_encoding 1 # --crop_mask 1 --refeed 0 \
    done
done