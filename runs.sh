#!/bin/sh
for seeds in 0 1 2
do
    for aldeghiFTPercentage in 0.01 0.02 0.04 0.1 0.2
    do
        #sbatch ./run.sbatch --seeds ${seeds} --pretrain.layer_norm ${layer_norm} --pretrain.early_stopping ${early_stopping} --finetune.aldeghiFTPercentage ${aldeghiFTPercentage}
        layer_norm=0
        early_stopping=0
        sbatch ./run.sbatch --seeds ${seeds} --pretrain.layer_norm ${layer_norm} --pretrain.early_stopping ${early_stopping} --finetune.diblockFTPercentage ${diblockFTPercentage}
    done
done
# Run this with addtional modifications in config:, i.e., PL true&false, Pretraining True and False, etc. 