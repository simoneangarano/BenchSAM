for p in sparsegpt
do
    for s in 0
    do
        python3 main.py --cuda 7 --dataset sa1b --model MobileSAM --edge_filter 1 --edge_width 20 \
                        --experiment filter20_ --pruning_method $p --sparsity $s --suffix _iou_bce_sigm \
                        --weights distilled_mobile_sam_decoder_iou_bce_15.pt --crop_mask 1 # --refeed 0  --sigmoid 1 \
    done
done